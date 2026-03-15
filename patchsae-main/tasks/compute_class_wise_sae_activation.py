import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.utils import get_model_activations, process_model_inputs
from tasks.utils import (
    DATASET_INFO,
    SAE_DIM,
    get_classnames,
    get_sae_and_vit,
    setup_save_directory,
    split_classnames,
)


def get_sae_activations_per_sample(
    model_activations: torch.Tensor, sae: SparseAutoencoder, threshold: float
) -> torch.Tensor:
    """Return per-sample SAE active-feature counts.

    Output shape is [batch, d_sae], where each value is the number of active tokens
    for that feature in the sample (or 0/1 for class-token SAE).
    """
    # run_with_cache captures internal SAE activations without modifying behavior.
    _, cache = sae.run_with_cache(model_activations)
    # Binary active mask for each latent (thresholded activation).
    # threshold=0.2 means "activation > 0.2 counts as active".
    activations = cache["hook_hidden_post"] > threshold

    if activations.ndim == 3:
        # [batch, token, d_sae] -> [batch, d_sae]
        # Count active tokens per latent for each sample.
        return activations.sum(dim=1).to(torch.int64)
    if activations.ndim == 2:
        # [batch, d_sae]
        # Class-token SAE path: activation already per sample per latent.
        return activations.to(torch.int64)

    raise ValueError(f"Unexpected SAE activation shape: {tuple(activations.shape)}")


def compute_all_class_activations_streaming(
    dataset,
    class_index_map: dict[int, int],
    sae: SparseAutoencoder,
    vit,
    cfg,
    batch_size: int,
    threshold: float,
    device: str,
) -> np.ndarray:
    """Compute class-wise SAE activation counts in one dataset pass.

    This avoids loading the entire dataset into Python lists by class.
    """
    # Rows = classes, cols = SAE latent dimensions.
    # Value = accumulated activation counts from all samples in that class.
    class_activation_counts = np.zeros((len(class_index_map), SAE_DIM), dtype=np.int64)
    total_iterations = (len(dataset) + batch_size - 1) // batch_size

    for iteration in tqdm(range(total_iterations)):
        # Slice one batch directly from Hugging Face dataset.
        batch_start = iteration * batch_size
        batch_end = min((iteration + 1) * batch_size, len(dataset))
        batch_dict = dataset[batch_start:batch_end]

        # Map original dataset labels -> local split labels.
        labels = np.asarray(batch_dict["label"], dtype=np.int64)
        mapped_labels = np.asarray(
            [class_index_map.get(int(label), -1) for label in labels], dtype=np.int64
        )
        keep_mask = mapped_labels >= 0

        # If this batch has no class from current split, skip.
        if not keep_mask.any():
            continue

        # Keep only samples that belong to selected class split.
        if keep_mask.all():
            filtered_batch = batch_dict
        else:
            keep_indices = np.nonzero(keep_mask)[0].tolist()
            # Rebuild the dict-of-lists batch with only selected rows.
            filtered_batch = {
                key: [batch_dict[key][index] for index in keep_indices]
                for key in batch_dict
            }
            mapped_labels = mapped_labels[keep_mask]

        batch_inputs = process_model_inputs(filtered_batch, vit, device)
        transformer_activations = get_model_activations(
            vit, batch_inputs, cfg.block_layer, cfg.module_name, cfg.class_token
        )
        active_features = (
            get_sae_activations_per_sample(transformer_activations, sae, threshold)
            .to(torch.int64)
            .cpu()
            .numpy()
        )

        # Add each sample's latent counts to its class row.
        # np.add.at supports repeated indices and performs safe in-place accumulation.
        np.add.at(class_activation_counts, mapped_labels, active_features)
        if torch.cuda.is_available():
            # This helps low-memory GPUs but can reduce throughput on strong GPUs.
            torch.cuda.empty_cache()

    return class_activation_counts


def main(
    sae_path: str,
    vit_type: str,
    device: str,
    dataset_name: str,
    root_dir: str,
    save_name: str,
    backbone: str = "openai/clip-vit-base-patch16",
    batch_size: int = 8,
    model_path: str = None,
    config_path: str = None,
    threshold: float = 0.2,
    split: str = "all",
):
    """Main function to compute and save class-wise SAE activation counts."""

    # Include split in output folder to avoid accidental overwrite.
    split_suffix = vit_type if split == "all" else f"{vit_type}_{split}"
    save_directory = setup_save_directory(
        root_dir, save_name, sae_path, split_suffix, dataset_name
    )

    dataset = load_dataset(**DATASET_INFO[dataset_name])
    classnames_all = get_classnames(dataset_name, dataset)
    classnames, selected_class_indices = split_classnames(classnames_all, split)
    # Original class index -> local row index in output matrix.
    # Example: if split="novel", local row 0 may correspond to original class 500.
    class_index_map = {
        int(class_index): local_index
        for local_index, class_index in enumerate(selected_class_indices)
    }

    sae, vit, cfg = get_sae_and_vit(
        sae_path,
        vit_type,
        device,
        backbone,
        model_path=model_path,
        config_path=config_path,
        classnames=classnames,
    )

    class_activation_counts = compute_all_class_activations_streaming(
        dataset, class_index_map, sae, vit, cfg, batch_size, threshold, device
    )

    # Save results as [num_selected_classes, SAE_DIM].
    save_path = os.path.join(save_directory, "cls_sae_cnt.npy")
    np.save(save_path, class_activation_counts)
    print(f"Class activation counts saved at {save_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save class-wise SAE activation count")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument(
        "--sae_path", type=str, required=True, help="SAE ckpt path (ends with xxx.pt)"
    )
    parser.add_argument(
        "--vit_type", type=str, default="base", help="choose between [base, maple]"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="threshold for SAE activation"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model_path",
        type=str,
        help="CLIP model path in the case of not using the default",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="CLIP config path in the case of using maple",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "base", "novel"],
        help="Class split for base-to-novel evaluation",
    )

    args = parser.parse_args()

    main(
        sae_path=args.sae_path,
        vit_type=args.vit_type,
        device=args.device,
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        save_name="out/feature_data",
        batch_size=args.batch_size,
        model_path=args.model_path,
        config_path=args.config_path,
        threshold=args.threshold,
        split=args.split,
    )
