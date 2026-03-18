from pathlib import Path

import torch
from datasets import load_dataset

from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)


_BUNDLE_REQUIRED_FILES = (
    "max_activating_image_indices.pt",
    "max_activating_image_values.pt",
    "sae_mean_acts.pt",
)
_BUNDLE_FEATURE_FILENAMES = {
    "max_activating_image_indices.pt",
    "max_activating_image_label_indices.pt",
    "max_activating_image_values.pt",
    "sae_mean_acts.pt",
    "sae_sparsity.pt",
}


def _resolve_maple_model_path() -> str:
    """Resolve default MaPLe checkpoint path across common repo layouts."""
    candidates = [
        "./data/clip/maple/imagenet/model.pth.tar-2",
        "./configs/models/maple/model.pth.tar-2",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return candidates[0]


def _resolve_sae_path(sae_path: str | None, feature_bundle_dir: str | None) -> str:
    """Resolve SAE checkpoint path.

    If feature_bundle_dir is provided and sae_path is empty, infer checkpoint file from bundle.
    """
    if sae_path is not None and str(sae_path).strip():
        candidate = Path(str(sae_path))
        if candidate.exists():
            return str(candidate)
        # In bundle mode, if explicit path is invalid, try auto-detection in bundle.
        if not feature_bundle_dir:
            raise FileNotFoundError(f"sae_path does not exist: {candidate}")

    if not feature_bundle_dir:
        raise ValueError("sae_path is required when feature_bundle_dir is not provided.")

    bundle = Path(feature_bundle_dir)
    candidates = [
        p for p in bundle.glob("*.pt") if p.name not in _BUNDLE_FEATURE_FILENAMES
    ]
    if len(candidates) == 1:
        return str(candidates[0])

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No SAE checkpoint found in bundle directory: {bundle}. "
            "Please pass --sae_path explicitly."
        )

    raise ValueError(
        f"Multiple checkpoint candidates found in {bundle}: {[p.name for p in candidates]}. "
        "Please pass --sae_path explicitly."
    )


def _load_bundle_feature_data(feature_bundle_dir: str) -> tuple[dict, dict]:
    """Load max-activating image indices and mean activations from a single bundle directory."""
    bundle = Path(feature_bundle_dir)
    missing = [name for name in _BUNDLE_REQUIRED_FILES if not (bundle / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Feature bundle is missing files: {missing}. directory={bundle}"
        )

    max_act_imgs = {
        "imagenet": torch.load(
            bundle / "max_activating_image_indices.pt", map_location="cpu"
        ).to(torch.int32)
    }
    mean_acts = {
        "imagenet": torch.load(bundle / "sae_mean_acts.pt", map_location="cpu").numpy()
    }
    return max_act_imgs, mean_acts


def _load_imagenet_only(seed: int = 1) -> dict:
    """Load ImageNet split only. Used by single-bundle demo mode."""
    return {
        "imagenet": load_dataset(
            "evanarlian/imagenet_1k_resized_256", split="train"
        ).shuffle(seed=seed)
    }


def load_sae_tester(
    sae_path: str | None,
    include_imagenet: bool = False,
    feature_bundle_dir: str | None = None,
):
    single_bundle_mode = bool(
        feature_bundle_dir is not None and str(feature_bundle_dir).strip()
    )
    sae_path = _resolve_sae_path(sae_path, feature_bundle_dir)

    if single_bundle_mode:
        # In bundle mode we only use ImageNet references shipped in the same folder.
        datasets = _load_imagenet_only(seed=1)
        classnames = get_all_classnames(datasets, data_root="./configs/classnames")
        max_act_imgs, mean_acts = _load_bundle_feature_data(feature_bundle_dir)
    else:
        datasets = load_datasets(include_imagenet=include_imagenet)
        classnames = get_all_classnames(datasets, data_root="./configs/classnames")

        root = "./out/feature_data"
        sae_runname = "sae_base"
        vit_name = "base"

        if include_imagenet is False:
            datasets["imagenet"] = None

        max_act_imgs, mean_acts = get_max_acts_and_images(
            datasets, root, sae_runname, vit_name
        )

    sae_tester = {}
    maple_model_path = _resolve_maple_model_path()

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="base",
        device="cpu",
        backbone="openai/clip-vit-base-patch16",
        model_path=None,
        classnames=None,
    )
    sae_clip = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="maple",
        device="cpu",
        model_path=maple_model_path,
        config_path="./configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml",
        backbone="openai/clip-vit-base-patch16",
        classnames=classnames["imagenet"],
    )
    sae_maple = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)
    sae_tester["CLIP"] = sae_clip
    sae_tester["MaPLE-imagenet"] = sae_maple
    return sae_tester
