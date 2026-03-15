import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.models.utils import get_adapted_clip, get_base_clip
from src.sae_training.config import Config
from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder

# Dataset registry used by task scripts.
# Note:
# - `imagenet` here points to `train` split of the resized HF version.
# - Changing split/path here will affect ALL tasks that import DATASET_INFO.
DATASET_INFO = {
    "imagenet": {
        "path": "evanarlian/imagenet_1k_resized_256",
        "split": "train",
    },
    "imagenet-sketch": {
        "path": "clip-benchmark/wds_imagenet_sketch",
        "split": "train",
    },
    "oxford_flowers": {
        "path": "nelorth/oxford-flowers",
        "split": "train",
    },
    "caltech101": {
        "path": "HuggingFaceM4/Caltech-101",
        "split": "train",
        "name": "with_background_category",
    },
}

SAE_DIM = 49152


def load_sae(sae_path: str, device: str) -> tuple[SparseAutoencoder, Config]:
    """Load a sparse autoencoder model from a checkpoint file."""
    # Always load checkpoint to CPU first, then move model to target device.
    # This is safer and avoids OOM during deserialization.
    checkpoint = torch.load(sae_path, map_location="cpu")

    # Backward compatibility: some checkpoints use "cfg", others "config".
    if "cfg" in checkpoint:
        cfg = Config(checkpoint["cfg"])
    else:
        cfg = Config(checkpoint["config"])

    # Build SAE from config and restore trained weights.
    # `cfg` defines SAE width, hook location, and training hyperparameters.
    sae = SparseAutoencoder(cfg, device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval().to(device)

    return sae, cfg


def load_hooked_vit(
    cfg: Config,
    vit_type: str,
    backbone: str,
    device: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> HookedVisionTransformer:
    """Load a vision transformer model with hooks."""
    # base -> original CLIP
    # maple -> adapted CLIP (prompt-learning style)
    if vit_type == "base":
        model, processor = get_base_clip(backbone)
    else:
        model, processor = get_adapted_clip(
            cfg, vit_type, model_path, config_path, backbone, classnames
        )

    # Wrap model with hook-friendly helper.
    return HookedVisionTransformer(model, processor, device=device)


def get_sae_and_vit(
    sae_path: str,
    vit_type: str,
    device: str,
    backbone: str,
    model_path: str = None,
    config_path: str = None,
    classnames: list[str] = None,
) -> tuple[SparseAutoencoder, HookedVisionTransformer, Config]:
    """Load both SAE and ViT models."""
    # Shared utility used by most task scripts to keep model loading consistent.
    # This prevents subtle mismatch (e.g., wrong backbone or wrong hook layer).
    sae, cfg = load_sae(sae_path, device)
    vit = load_hooked_vit(
        cfg, vit_type, backbone, device, model_path, config_path, classnames
    )
    return sae, vit, cfg


def load_and_organize_dataset(dataset_name: str) -> Tuple[list, Dict]:
    # TODO: ERR for imagenet (gets killed after 75%)
    """
    Load dataset and organize data by class.
    Return classnames and data by class.
    Requried for classification_with_top_k_masking.py and compute_class_wise_sae_activation.py
    """
    dataset = load_dataset(**DATASET_INFO[dataset_name])
    classnames = get_classnames(dataset_name, dataset)

    # Build classname -> list[sample] dictionary.
    # This is easy to use but memory-heavy on large datasets.
    data_by_class = defaultdict(list)
    for data_item in tqdm(dataset):
        classname = classnames[data_item["label"]]
        # Each value is a full sample dict from Hugging Face dataset.
        data_by_class[classname].append(data_item)

    return classnames, data_by_class


def split_classnames(classnames: list[str], split: str) -> tuple[list[str], list[int]]:
    """Split class names into base/novel subsets by class index order.

    The split follows the paper protocol: first half is base, remaining is novel.
    """
    split = split.lower()
    if split not in {"all", "base", "novel"}:
        raise ValueError(f"Unsupported split: {split}. Use one of all/base/novel.")

    if split == "all":
        indices = list(range(len(classnames)))
    else:
        # Paper convention: first half "base", second half "novel".
        midpoint = len(classnames) // 2
        if split == "base":
            indices = list(range(midpoint))
        else:
            indices = list(range(midpoint, len(classnames)))

    split_names = [classnames[i] for i in indices]
    return split_names, indices


def filter_data_by_split(
    classnames: list[str], data_by_class: Dict, split: str
) -> tuple[list[str], Dict]:
    """Filter class-wise dataset dictionary with the selected class split."""
    split_names, _ = split_classnames(classnames, split)
    filtered_data = {classname: data_by_class[classname] for classname in split_names}
    return split_names, filtered_data


def get_classnames(
    dataset_name: str, dataset: Dataset = None, data_root: str = "./configs/classnames"
) -> list[str]:
    """Get class names for a dataset."""

    # Classname files are stored in configs/classnames.
    # Supports either txt or json depending on dataset.
    filename = f"{data_root}/{dataset_name}_classnames"
    txt_filename = filename + ".txt"
    json_filename = filename + ".json"

    if not os.path.exists(txt_filename) and not os.path.exists(json_filename):
        raise ValueError(f"Dataset {dataset_name} not supported")

    filename = json_filename if os.path.exists(json_filename) else txt_filename

    with open(filename, "r") as file:
        if dataset_name == "caltech101":
            # Caltech file already stores plain class names per line.
            class_names = [line.strip() for line in file.readlines()]
        elif dataset_name == "imagenet" or dataset_name == "imagenet-sketch":
            # ImageNet file format usually begins with synset id; drop it.
            class_names = [
                " ".join(line.strip().split(" ")[1:]) for line in file.readlines()
            ]
        elif dataset_name == "oxford_flowers":
            # Oxford flowers classnames file is keyed by HF class names.
            assert dataset is not None, "Dataset must be provided for Oxford Flowers"
            new_class_dict = {}
            class_names = json.load(file)
            classnames_from_hf = dataset.features["label"].names
            for i, class_name in enumerate(classnames_from_hf):
                new_class_dict[i] = class_names[class_name]
            class_names = list(new_class_dict.values())

        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    return class_names


def setup_save_directory(
    root_dir: str, save_name: str, sae_path: str, vit_type: str, dataset_name: str
) -> str:
    """Set and create the save directory path."""
    if not sae_path or not str(sae_path).strip():
        raise ValueError(
            "sae_path is empty. Please pass a valid checkpoint path to --sae_path."
        )

    # Normalize separators for Windows/Linux compatibility.
    normalized_path = str(sae_path).replace("\\", "/")
    path_parts = [part for part in normalized_path.split("/") if part]

    # Usually use parent folder as run name; fallback to filename stem.
    # Example:
    # sae_path=/.../final_sparse_autoencoder_openai/clip-vit...pt
    # sae_run_name becomes "final_sparse_autoencoder_openai"
    if len(path_parts) >= 2:
        sae_run_name = path_parts[-2]
    else:
        sae_run_name = os.path.splitext(path_parts[-1])[0]

    # Canonical output layout used by all downstream scripts.
    save_directory = (
        f"{root_dir}/{save_name}/sae_{sae_run_name}/{vit_type}/{dataset_name}"
    )
    os.makedirs(save_directory, exist_ok=True)
    return save_directory


def get_sae_activations(
    model_activations: torch.Tensor, sae: SparseAutoencoder
) -> torch.Tensor:
    """Extract and process activations from the sparse autoencoder."""
    hook_name = "hook_hidden_post"

    # Run SAE forward pass and get hidden-post activations from hook cache.
    _, cache = sae.run_with_cache(model_activations)
    sae_activations = cache[hook_name]

    # If token dimension exists, average over tokens to get per-image activation.
    if len(sae_activations.size()) > 2:
        # [batch, seq, d_sae] -> [batch, d_sae]
        sae_activations = sae_activations.mean(dim=1)

    return sae_activations


def process_batch(vit, batch_data, device):
    """Process a single batch of images."""
    # Convert list of sample dicts into processor-ready image list.
    images = [data["image"] for data in batch_data]

    # Keep text field for consistent CLIP input schema.
    inputs = vit.processor(
        images=images, text="", return_tensors="pt", padding=True
    ).to(device)
    return inputs


def get_max_acts_and_images(
    datasets: dict, feat_data_root: str, sae_runname: str, vit_name: str
) -> tuple[dict, dict]:
    """Load and return maximum activations and mean activations for each dataset."""
    max_act_imgs = {}
    mean_acts = {}

    for dataset_name in datasets:
        # Load max activating image indices
        max_act_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "max_activating_image_indices.pt",
        )
        max_act_imgs[dataset_name] = torch.load(max_act_path, map_location="cpu").to(
            torch.int32
        )
        # Shape is typically [d_sae, top_n].

        # Load mean activations
        mean_acts_path = os.path.join(
            feat_data_root,
            f"{sae_runname}/{vit_name}/{dataset_name}",
            "sae_mean_acts.pt",
        )
        mean_acts[dataset_name] = torch.load(mean_acts_path, map_location="cpu").numpy()
        # Shape is [d_sae], one scalar summary per latent.

    return max_act_imgs, mean_acts


def load_datasets(include_imagenet: bool = False, seed: int = 1):
    """Load multiple datasets from HuggingFace."""
    # Used by demo/evaluation helpers; all datasets are shuffled for randomness.
    if include_imagenet:
        return {
            "imagenet": load_dataset(
                "evanarlian/imagenet_1k_resized_256", split="train"
            ).shuffle(seed=seed),
            "imagenet-sketch": load_dataset(
                "clip-benchmark/wds_imagenet_sketch", split="test"
            ).shuffle(seed=seed),
            "caltech101": load_dataset(
                "HuggingFaceM4/Caltech-101",
                "with_background_category",
                split="train",
            ).shuffle(seed=seed),
        }
    else:
        return {
            "imagenet-sketch": load_dataset(
                "clip-benchmark/wds_imagenet_sketch", split="test"
            ).shuffle(seed=seed),
            "caltech101": load_dataset(
                "HuggingFaceM4/Caltech-101",
                "with_background_category",
                split="train",
            ).shuffle(seed=seed),
        }


def get_all_classnames(datasets, data_root):
    """Get class names for all datasets."""
    class_names = {}
    for dataset_name, dataset in datasets.items():
        class_names[dataset_name] = get_classnames(dataset_name, dataset, data_root)

    # Keep ImageNet class names available because MaPLe helper may rely on them.
    if "imagenet" not in class_names:
        # Fallback: load ImageNet names explicitly for methods needing shared label space.
        filename = f"{data_root}/imagenet_classnames"
        txt_filename = filename + ".txt"
        json_filename = filename + ".json"

        if not os.path.exists(txt_filename) and not os.path.exists(json_filename):
            raise ValueError(f"Dataset {dataset_name} not supported")

        filename = json_filename if os.path.exists(json_filename) else txt_filename

        with open(filename, "r") as file:
            class_names["imagenet"] = [
                " ".join(line.strip().split(" ")[1:]) for line in file.readlines()
            ]

    return class_names
