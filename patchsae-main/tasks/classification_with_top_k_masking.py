import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.models.templates.openai_imagenet_templates import openai_imagenet_template
from src.sae_training.config import Config
from src.sae_training.hooked_vit import Hook, HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.utils import process_model_inputs
from tasks.utils import (
    DATASET_INFO,
    SAE_DIM,
    get_classnames,
    get_sae_and_vit,
    setup_save_directory,
    split_classnames,
)

# Top-k latent sets used in the paper-style masking ablation.
# For each k, this script runs both:
#   - on_k: keep only the class-selected top-k SAE latents
#   - off_k: suppress exactly those top-k latents
# plus one baseline "no_sae" run.
# The final value `SAE_DIM` means "use all SAE latents".
TOPK_LIST = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, SAE_DIM]
SAE_BIAS = -0.105131256516992


def calculate_text_features(model, device, classnames):
    """Calculate mean text features across templates for each class."""
    # We average CLIP text embeddings over multiple prompt templates
    # (e.g., "a photo of a {class}") to reduce prompt sensitivity.
    # Start from scalar 0; after first addition this becomes a tensor accumulator.
    mean_text_features = 0

    for template_fn in openai_imagenet_template:
        # Generate prompts and convert to token IDs
        prompts = [template_fn(c) for c in classnames]
        # CLIP tokenizer returns variable-length token IDs per prompt.
        prompt_ids = [
            model.processor(
                text=p, return_tensors="pt", padding=False, truncation=True
            ).input_ids[0]
            for p in prompts
        ]

        # Process batch
        padded_prompts = pad_sequence(prompt_ids, batch_first=True).to(device)

        # Get text features
        with torch.no_grad():
            text_features = model.model.get_text_features(padded_prompts)
            # Normalize so cosine similarity can be computed by dot product.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features += text_features

    # Final class text bank: [num_classes, embed_dim]
    return mean_text_features / len(openai_imagenet_template)


def create_sae_hooks(vit_type, cfg, cls_features, sae, device, hook_type="on"):
    """Create SAE hooks based on model type and hook type."""
    # Setup clamping parameters
    # True for all latent dims (we clamp on full SAE latent space, then
    # selectively keep/suppress class-specific dims via clamp_value).
    clamp_feat_dim = torch.ones(SAE_DIM).bool()
    clamp_value = torch.zeros(SAE_DIM) if hook_type == "on" else torch.ones(SAE_DIM)
    clamp_value = clamp_value.to(device)
    # on: selected dims -> 1, others -> 0
    # off: selected dims -> 0, others -> 1
    clamp_value[cls_features] = 1.0 if hook_type == "on" else 0.0

    def process_activations(activations, is_maple=False):
        """Helper function to process activations with SAE"""
        # MaPLe hooks use a different tensor layout, so we transpose when needed.
        act = activations.transpose(0, 1) if is_maple else activations
        processed = (
            # forward_clamp returns a tuple; index 0 is reconstructed activations.
            sae.forward_clamp(
                act[:, :, :], clamp_feat_dim=clamp_feat_dim, clamp_value=clamp_value
            )[0]
            - SAE_BIAS
        )
        # Keep original dtype to avoid Float/Half mismatches in MaPLe attention layers.
        return processed.to(act.dtype)

    def hook_fn_default(activations):
        # In-place replacement of block activations for CLIP/base model.
        activations[:, :, :] = process_activations(activations)
        return (activations,)

    def hook_fn_maple(activations):
        # MaPLe expects transposed output shape.
        activations = process_activations(activations, is_maple=True)
        return activations.transpose(0, 1)

    # Create appropriate hook based on model type
    hook_fn = hook_fn_maple if vit_type == "maple" else hook_fn_default
    is_custom = vit_type == "maple"

    return [
        Hook(
            cfg.block_layer,
            cfg.module_name,
            hook_fn,
            return_module_output=False,
            is_custom=is_custom,
        )
    ]


def get_predictions(vit, inputs, text_features, vit_type, hooks=None):
    """Get model predictions with optional hooks."""
    with torch.no_grad():
        # Either run original model (no hooks) or patched model (with SAE hook).
        if hooks:
            vit_out = vit.run_with_hooks(hooks, return_type="output", **inputs)
        else:
            vit_out = vit(return_type="output", **inputs)

        # base: CLIPModel output object -> use image_embeds
        # maple: code path already returns image features tensor
        image_features = vit_out.image_embeds if vit_type == "base" else vit_out
        logit_scale = vit.model.logit_scale.exp()
        # Since text features are normalized, this is scaled cosine logits.
        logits = logit_scale * image_features @ text_features.t()
        preds = logits.argmax(dim=-1)

    return preds.cpu().numpy().tolist()


def build_class_to_indices(dataset, selected_class_indices: list[int]) -> list[list[int]]:
    """Build class -> sample-index mapping without storing full image objects."""
    # Memory-efficient indexing:
    # keep only integer indices per class, avoid storing PIL images in RAM.
    class_to_indices = [[] for _ in range(len(selected_class_indices))]
    original_to_local = {
        int(original_class_idx): local_idx
        for local_idx, original_class_idx in enumerate(selected_class_indices)
    }

    labels = dataset["label"]
    for sample_idx, label in enumerate(
        tqdm(labels, desc="Indexing dataset by class", leave=False)
    ):
        local_idx = original_to_local.get(int(label))
        if local_idx is not None:
            class_to_indices[local_idx].append(sample_idx)

    return class_to_indices


def classify_with_top_k_masking(
    dataset,
    class_indices: list[int],
    cls_idx: int,
    class_name: str,
    sae: SparseAutoencoder,
    vit: HookedVisionTransformer,
    cls_sae_cnt: torch.Tensor,
    text_features: torch.Tensor,
    batch_size: int,
    device: str,
    vit_type: str,
    cfg: Config,
    show_inner_progress: bool = False,
):
    """Classify images with top-k feature masking."""
    # Inner progress length is number of batches for THIS class.
    num_batches = (len(class_indices) + batch_size - 1) // batch_size

    preds_dict = defaultdict(list)
    # Pre-sort SAE latent importances for this class once.
    loaded_cls_sae_idx = cls_sae_cnt[cls_idx].argsort()[::-1]
    # Cost note:
    # per batch we run 1 baseline + (2 * len(TOPK_LIST)) hooked forward passes.
    # This is why this stage is much slower than plain CLIP inference.

    batch_iter = range(num_batches)
    if show_inner_progress:
        batch_iter = tqdm(
            batch_iter,
            total=num_batches,
            desc=f"class {cls_idx} ({class_name})",
            leave=False,
        )

    for batch_idx in batch_iter:
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(class_indices))
        batch_sample_indices = class_indices[batch_start:batch_end]
        # Hugging Face dataset supports list-based indexing and returns dict-of-lists.
        batch_dict = dataset[batch_sample_indices]
        batch_inputs = process_model_inputs(batch_dict, vit, device)

        # 1) Baseline prediction without intervention.
        preds_dict["no_sae"].extend(
            get_predictions(vit, batch_inputs, text_features, vit_type)
        )
        torch.cuda.empty_cache()

        # 2) For each k, run on/off interventions.
        for topk in TOPK_LIST:
            cls_features = loaded_cls_sae_idx[:topk].tolist()

            # Get predictions with features ON
            hooks_on = create_sae_hooks(vit_type, cfg, cls_features, sae, device, "on")
            preds_dict[f"on_{topk}"].extend(
                get_predictions(vit, batch_inputs, text_features, vit_type, hooks_on)
            )
            torch.cuda.empty_cache()

            # Get predictions with features OFF
            hooks_off = create_sae_hooks(
                vit_type, cfg, cls_features, sae, device, "off"
            )
            preds_dict[f"off_{topk}"].extend(
                get_predictions(vit, batch_inputs, text_features, vit_type, hooks_off)
            )
            torch.cuda.empty_cache()

    return preds_dict


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
    cls_wise_sae_activation_path: str = None,
    split: str = "all",
    show_inner_progress: bool = False,
):
    # Save suffix indicates which cls-wise activation source and split are used.
    class_feature_type = cls_wise_sae_activation_path.split("/")[-3]
    save_suffix = (
        f"{class_feature_type}_{vit_type}"
        if split == "all"
        else f"{class_feature_type}_{vit_type}_{split}"
    )
    save_directory = setup_save_directory(
        root_dir, save_name, sae_path, save_suffix, dataset_name
    )

    dataset = load_dataset(**DATASET_INFO[dataset_name])
    classnames_all = get_classnames(dataset_name, dataset)
    classnames, selected_class_indices = split_classnames(classnames_all, split)
    # Build class -> index list once; later batches fetch by index slices.
    # This avoids holding full sample dicts in memory for each class.
    class_to_indices = build_class_to_indices(dataset, selected_class_indices)

    sae, vit, cfg = get_sae_and_vit(
        sae_path,
        vit_type,
        device,
        backbone,
        model_path=model_path,
        config_path=config_path,
        classnames=classnames,
    )

    cls_sae_cnt = np.load(cls_wise_sae_activation_path)
    if cls_sae_cnt.shape[0] != len(classnames):
        raise ValueError(
            "Mismatch between class split and cls_sae_cnt rows. "
            f"Got {cls_sae_cnt.shape[0]} rows, expected {len(classnames)} "
            f"for split='{split}'."
        )

    if vit_type == "base":
        # CLIP baseline: derive text prototypes from prompt templates.
        text_features = calculate_text_features(vit, device, classnames)
    else:
        # MaPLe path already stores/serves text features in model object.
        text_features = vit.model.get_text_features()

    metrics_dict = {}
    for class_idx, classname in enumerate(tqdm(classnames)):
        # For each class, evaluate all intervention settings and collect predictions.
        # NOTE: class_idx here is local index within selected split.
        preds_dict = classify_with_top_k_masking(
            dataset,
            class_to_indices[class_idx],
            class_idx,
            classname,
            sae,
            vit,
            cls_sae_cnt,
            text_features,
            batch_size,
            device,
            vit_type,
            cfg,
            show_inner_progress=show_inner_progress,
        )

        metrics_dict[class_idx] = {}
        for k, v in preds_dict.items():
            # Per-class top-1 accuracy (%), where "correct" means predicting class_idx.
            metrics_dict[class_idx][k] = (
                v.count(class_idx) / len(v) * 100 if len(v) > 0 else 0.0
            )

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(f"{save_directory}/metrics.csv", index=False)
    print(f"metrics.csv saved at {save_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform classification with top-k masking"
    )
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument(
        "--sae_path", type=str, required=True, help="SAE ckpt path (ends with xxx.pt)"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--cls_wise_sae_activation_path",
        type=str,
        help="path for cls_sae_cnt.npy",
    )
    parser.add_argument(
        "--vit_type", type=str, default="base", help="choose between [base, maple]"
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
        "--split",
        type=str,
        default="all",
        choices=["all", "base", "novel"],
        help="Class split for base-to-novel evaluation",
    )
    parser.add_argument(
        "--show_inner_progress",
        action="store_true",
        default=False,
        help="Show per-class batch-level progress bars.",
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
        cls_wise_sae_activation_path=args.cls_wise_sae_activation_path,
        split=args.split,
        show_inner_progress=args.show_inner_progress,
    )
