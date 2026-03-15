import argparse
from typing import Dict, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.utils import get_model_activations, process_model_inputs
from tasks.utils import (
    DATASET_INFO,
    get_classnames,
    get_sae_activations,
    get_sae_and_vit,
    setup_save_directory,
)


def initialize_storage_tensors(
    d_sae: int, num_max: int, device: str
) -> Dict[str, torch.Tensor]:
    """Initialize tensors for storing results."""
    # max_activating_image_values/indices:
    #   For each SAE latent, keep top-N highest activation image IDs.
    # sae_sparsity:
    #   Running count of how many samples activate each latent (>0).
    # sae_mean_acts:
    #   Running sum of activation values for each latent.
    return {
        "max_activating_image_values": torch.zeros([d_sae, num_max]).to(device),
        "max_activating_image_indices": torch.zeros(
            [d_sae, num_max], dtype=torch.long, device=device
        ),
        "sae_sparsity": torch.zeros([d_sae]).to(device),
        "sae_mean_acts": torch.zeros([d_sae]).to(device),
    }


def get_new_top_k(
    first_values: torch.Tensor,
    first_indices: torch.Tensor,
    second_values: torch.Tensor,
    second_indices: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top k values and indices from two sets of values/indices."""
    # Merge "existing top-k" and "current-batch top-k", then keep global top-k.
    # Shapes:
    #   first_values/second_values: [d_sae, k]
    #   output new_values/new_indices: [d_sae, k]
    total_values = torch.cat([first_values, second_values], dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, indices_of_indices = torch.topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices


def compute_sae_statistics(
    sae_activations: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean activations and sparsity statistics for SAE features."""
    # sae_activations shape: [d_sae, batch]
    # mean_acts is actually "sum of activations over batch" and is normalized later.
    mean_acts = sae_activations.sum(dim=1)
    sparsity = (sae_activations > 0).sum(dim=1)
    return mean_acts, sparsity


def get_top_activations(
    sae_activations: torch.Tensor, num_top_images: int, images_processed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top activating images and their indices."""
    # top_k cannot exceed current batch size.
    top_k = min(num_top_images, sae_activations.size(1))
    values, indices = torch.topk(sae_activations, k=top_k, dim=1)
    # Convert batch-local indices to dataset-global indices.
    indices += images_processed
    return values, indices


def process_batch(
    batch: Dict,
    vit: HookedVisionTransformer,
    sae: SparseAutoencoder,
    cfg: Config,
    device: str,
    num_top_images: int,
    images_processed: int,
    storage: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Process a single batch of images and update feature statistics."""
    # 1) Image -> ViT activations -> SAE latent activations.
    # `batch` is a Hugging Face dict-of-lists (not a PyTorch tensor batch).
    inputs = process_model_inputs(batch, vit, device)
    model_acts = get_model_activations(
        vit, inputs, cfg.block_layer, cfg.module_name, cfg.class_token
    )
    # get_sae_activations returns [batch, d_sae]; transpose -> [d_sae, batch]
    sae_acts = get_sae_activations(model_acts, sae).transpose(0, 1)

    # 2) Update running sums/counters.
    mean_acts, sparsity = compute_sae_statistics(sae_acts)
    storage["sae_mean_acts"] += mean_acts
    storage["sae_sparsity"] += sparsity

    # 3) Update per-latent global top-N images.
    values, indices = get_top_activations(sae_acts, num_top_images, images_processed)

    top_values, top_indices = get_new_top_k(
        storage["max_activating_image_values"],
        storage["max_activating_image_indices"],
        values,
        indices,
        num_top_images,
    )

    # 4) Update how many images have been processed so far.
    # Use raw batch image count (not activation tensor shape), so class-token
    # mode and backbone-specific tensor layouts cannot skew global indices.
    images_processed += len(batch["image"])

    return {
        "max_activating_image_values": top_values,
        "max_activating_image_indices": top_indices,
        "sae_sparsity": storage["sae_sparsity"],
        "sae_mean_acts": storage["sae_mean_acts"],
    }, images_processed


def save_results(
    save_directory: str,
    storage: Dict[str, torch.Tensor],
    dataset: Dataset,
    label_field: Optional[str] = None,
) -> None:
    """Save results to disk."""
    # Optional label lookup for visualization/debug.
    # This can be slow because it dereferences dataset rows one by one.
    if label_field and label_field in dataset.features:
        flat_indices = storage["max_activating_image_indices"].flatten()
        # Safety guard against unexpected out-of-range indices.
        flat_indices = flat_indices.clamp(min=0, max=len(dataset) - 1).to(torch.long)
        max_activating_image_label_indices = torch.tensor(
            [
                dataset[int(index)][label_field]
                for index in tqdm(
                    flat_indices,
                    desc="getting image labels",
                )
            ]
        ).view(storage["max_activating_image_indices"].shape)

        torch.save(
            max_activating_image_label_indices,
            f"{save_directory}/max_activating_image_label_indices.pt",
        )

    torch.save(
        storage["max_activating_image_indices"],
        f"{save_directory}/max_activating_image_indices.pt",
    )
    torch.save(
        storage["max_activating_image_values"],
        f"{save_directory}/max_activating_image_values.pt",
    )
    torch.save(storage["sae_sparsity"], f"{save_directory}/sae_sparsity.pt")
    torch.save(storage["sae_mean_acts"], f"{save_directory}/sae_mean_acts.pt")

    # Files saved:
    # - max_activating_image_indices.pt : [d_sae, top_n] image ids
    # - max_activating_image_values.pt  : [d_sae, top_n] activation values
    # - sae_sparsity.pt                 : [d_sae] activation frequency
    # - sae_mean_acts.pt                : [d_sae] mean activation on active samples
    print(f"Results saved to {save_directory}")


@torch.inference_mode()
def main(
    sae_path: str,
    vit_type: str,
    device: str,
    dataset_name: str,
    root_dir: str,
    save_name: str,
    backbone: str = "openai/clip-vit-base-patch16",
    number_of_max_activating_images: int = 10,
    seed: int = 1,
    batch_size: int = 8,
    model_path: str = None,
    config_path: str = None,
):
    """Main function to extract and save feature data."""
    # Use better fp32 matmul kernel when available.
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()

    # Shuffle only affects tie-break style of top-activating images.
    # Aggregate statistics (means/sparsity) remain distributional properties.
    dataset = load_dataset(**DATASET_INFO[dataset_name])
    dataset = dataset.shuffle(seed=seed)
    classnames = get_classnames(dataset_name, dataset)

    sae, vit, cfg = get_sae_and_vit(
        sae_path, vit_type, device, backbone, model_path, config_path, classnames
    )

    storage = initialize_storage_tensors(
        sae.cfg.d_sae, number_of_max_activating_images, device
    )

    # Process full dataset in batches and accumulate statistics.
    total_iterations = (len(dataset) + batch_size - 1) // batch_size
    # Number of samples actually consumed so far.
    num_processed = 0

    for iteration in tqdm(range(total_iterations)):
        batch_start = iteration * batch_size
        batch_end = (iteration + 1) * batch_size
        current_batch = dataset[batch_start:batch_end]

        storage, num_processed = process_batch(
            current_batch,
            vit,
            sae,
            cfg,
            device,
            number_of_max_activating_images,
            num_processed,
            storage,
        )

    # Convert running sums/counters into final metrics.
    # Note: sae_mean_acts is divided by activation count per latent.
    # If a latent is never active, division can produce inf/nan for that latent.
    # This is expected in sparse settings and can be handled downstream if needed.
    storage["sae_mean_acts"] /= storage["sae_sparsity"]
    # sae_sparsity becomes activation frequency per latent over all samples.
    storage["sae_sparsity"] /= num_processed

    # Save final tensors to the standardized output directory.
    save_directory = setup_save_directory(
        root_dir, save_name, sae_path, vit_type, dataset_name
    )
    save_results(
        save_directory,
        storage,
        dataset,
        label_field="label" if "label" in dataset.features else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ViT SAE images and save feature data"
    )
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument(
        "--sae_path", type=str, required=True, help="SAE ckpt path (ends with xxx.pt)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to compute model activations and sae features",
    )
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
        "--vit_type", type=str, default="base", help="choose between [base, maple]"
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
    )
