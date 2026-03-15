import math
from typing import Dict, Optional

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.sae_training.hooked_vit import HookedVisionTransformer

SAE_DIM = 49152


def process_model_inputs(
    batch: Dict, vit: HookedVisionTransformer, device: str, process_labels: bool = False
) -> torch.Tensor:
    """Process input images through the ViT processor."""
    # When process_labels=True, build class prompts from labels.
    # This mode is mainly for tasks that need image-text paired inputs.
    if process_labels:
        labels = [f"A photo of a {label}" for label in batch["label"]]
        return vit.processor(
            images=batch["image"], text=labels, return_tensors="pt", padding=True
        ).to(device)

    # Default mode: image-only workload, while still passing a dummy text field
    # to keep CLIP processor output format consistent with model forward.
    # Returned object typically contains:
    # - pixel_values: [batch, 3, H, W]
    # - input_ids / attention_mask for text branch (can be minimal here)
    return vit.processor(
        images=batch["image"], text="", return_tensors="pt", padding=True
    ).to(device)


def get_model_activations(
    model: HookedVisionTransformer, inputs: dict, block_layer, module_name, class_token
) -> torch.Tensor:
    """Extract activations from a specific layer of the vision transformer model."""
    hook_location = (block_layer, module_name)

    # Register one hook target and run a single forward pass.
    # Returned cache key is (block_layer, module_name).
    _, cache = model.run_with_cache([hook_location], **inputs)
    activations = cache[hook_location]

    # Different backbones may output [seq, batch, dim] or [batch, seq, dim].
    # Normalize to [batch, seq, dim] for downstream SAE code.
    batch_size = inputs["pixel_values"].shape[0]
    if activations.shape[0] != batch_size:
        activations = activations.transpose(0, 1)

    # If class_token mode is enabled, keep only CLS token activations:
    # [batch, seq, dim] -> [batch, dim]
    # This reduces memory and focuses on global image representation.
    if class_token:
        activations = activations[:, 0, :]

    return activations


def get_scheduler(scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs):
    # Linear warmup then linear decay to 0.
    def get_warmup_lambda(warm_up_steps, training_steps):
        def lr_lambda(steps):
            # Warmup phase: increase LR from near 0 to base LR.
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                # Decay phase: linearly reduce to 0 at final step.
                return (training_steps - steps) / (training_steps - warm_up_steps)

        return lr_lambda

    # Linear warmup then cosine decay to lr_end.
    def get_warmup_cosine_lambda(warm_up_steps, training_steps, lr_end):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                # progress in [0, 1], then cosine interpolation to lr_end.
                progress = (steps - warm_up_steps) / (training_steps - warm_up_steps)
                return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))

        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 500)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )
    elif scheduler_name.lower() == "linearwarmupdecay":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        lr_lambda = get_warmup_lambda(warm_up_steps, training_steps)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealing":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_steps, eta_min=eta_min
        )
    elif scheduler_name.lower() == "cosineannealingwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        lr_lambda = get_warmup_cosine_lambda(warm_up_steps, training_steps, eta_min)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealingwarmrestarts":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        num_cycles = kwargs.get("num_cycles", 1)
        # One restart period length.
        # Example: training_steps=10000, num_cycles=2 => restart every 5000 steps.
        T_0 = training_steps // num_cycles
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
