"""
# Portions of this file are based on code from the "HugoFry/mats_sae_training_for_ViTs" repository (MIT-licensed):
    https://github.com/HugoFry/mats_sae_training_for_ViTs/blob/main/sae_training/hooked_vit.py
"""

from contextlib import contextmanager
from functools import partial
from typing import Callable, List, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F
from transformers import CLIPModel


# The Hook class does not currently only supports hooking on the following locations:
# 1 - residual stream post transformer block.
# 2 - mlp activations.
# More hooks can be added at a later date, but only post-module.
# In this repository, we mainly use "resid" hooks for SAE-based interventions.
class Hook:
    def __init__(
        self,
        block_layer: int,
        module_name: str,
        hook_fn: Callable,
        is_custom: bool = None,
        return_module_output=True,
    ):
        self.path_dict = {
            # Empty suffix means hook module output of the full transformer block.
            "resid": "",
        }
        assert module_name in self.path_dict.keys(), (
            f"Module name '{module_name}' not recognised."
        )
        self.return_module_output = return_module_output
        # The adapter wraps user hook_fn into the signature expected by PyTorch hooks.
        self.function = self.get_full_hook_fn(hook_fn)
        self.attr_path = self.get_attr_path(block_layer, module_name, is_custom)

    def get_full_hook_fn(self, hook_fn: Callable):
        def full_hook_fn(module, module_input, module_output):
            # For CLIP blocks, output is typically a tuple whose first element is activations.
            # For MaPLe blocks, output is a list: [activations, compound_prompts_deeper, counter].
            if isinstance(module_output, (tuple, list)) and len(module_output) > 0:
                activations = module_output[0]
            else:
                activations = module_output

            hook_fn_output = hook_fn(activations)

            # Backward compatibility: some existing hooks return `(activations,)`.
            if isinstance(hook_fn_output, (tuple, list)) and len(hook_fn_output) == 1:
                hook_fn_output = hook_fn_output[0]

            if self.return_module_output:
                # Keep original module output unchanged.
                return module_output

            # When module output is composite (tuple/list), replace only the first
            # activation tensor and keep the auxiliary state (e.g., MaPLe counter).
            if isinstance(module_output, tuple):
                if len(module_output) == 0:
                    return module_output
                return (hook_fn_output, *module_output[1:])

            if isinstance(module_output, list):
                if len(module_output) == 0:
                    return module_output
                output = list(module_output)
                output[0] = hook_fn_output
                return output

            # Plain tensor output: directly replace with hooked activations.
            return hook_fn_output

        return full_hook_fn

    def get_attr_path(
        self, block_layer: int, module_name: str, is_custom: bool = None
    ) -> str:
        # CLIP path vs MaPLe-custom path differ in module nesting.
        if is_custom:
            attr_path = f"image_encoder.transformer.resblocks[{block_layer}]"
        else:
            attr_path = f"vision_model.encoder.layers[{block_layer}]"
        attr_path += self.path_dict[module_name]
        return attr_path

    def get_module(self, model):
        # Resolve a string path like "vision_model.encoder.layers[11]".
        return self.get_nested_attr(model, self.attr_path)

    def get_nested_attr(self, model, attr_path):
        """
        Gets a nested attribute from an object using a dot-separated path.
        """
        module = model
        attributes = attr_path.split(".")
        for attr in attributes:
            if "[" in attr:
                # Split at '[' and remove the trailing ']' from the index
                attr_name, index = attr[:-1].split("[")
                module = getattr(module, attr_name)[int(index)]
            else:
                module = getattr(module, attr)
        return module


class HookedVisionTransformer:
    def __init__(self, model, processor, device="cuda"):
        # Keep model and processor together for convenience in task scripts.
        # - model: CLIP or adapted CLIP-like model
        # - processor: tokenizer + image preprocessor bundle
        self.model = model.to(device)
        self.processor = processor

    def run_with_cache(
        self,
        list_of_hook_locations: List[Tuple[int, str]],
        *args,
        return_type="output",
        **kwargs,
    ):
        # Build runtime hooks, run one forward pass, and return cached activations.
        cache_dict, list_of_hooks = self.get_caching_hooks(list_of_hook_locations)
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                output = hooked_model(*args, **kwargs)

        if return_type == "output":
            # Return model output object + activations captured by hooks.
            return output, cache_dict
        if return_type == "loss":
            return (
                self.contrastive_loss(output.logits_per_image, output.logits_per_text),
                cache_dict,
            )
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def get_caching_hooks(self, list_of_hook_locations: List[Tuple[int, str]]):
        """
        Note that the cache dictionary is index by the tuple (block_layer, module_name).
        """
        cache_dict = {}
        list_of_hooks = []

        def save_activations(name, activations):
            # Detach to avoid keeping computation graph in memory.
            cache_dict[name] = activations.detach()

        for block_layer, module_name in list_of_hook_locations:
            hook_fn = partial(save_activations, (block_layer, module_name))
            if isinstance(self.model, CLIPModel):
                is_custom = False
            else:
                is_custom = True
            # "is_custom" controls how module path is resolved (CLIP vs MaPLe style).
            hook = Hook(block_layer, module_name, hook_fn, is_custom=is_custom)
            list_of_hooks.append(hook)
        return cache_dict, list_of_hooks

    @torch.no_grad()
    def run_with_hooks(
        self, list_of_hooks: List[Hook], *args, return_type="output", **kwargs
    ):
        # Run a forward pass with user-provided hooks (no cache returned).
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                output = hooked_model(*args, **kwargs)
        if return_type == "output":
            return output
        if return_type == "loss":
            return self.contrastive_loss(
                output.logits_per_image, output.logits_per_text
            )
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def train_with_hooks(
        self, list_of_hooks: List[Hook], *args, return_type="output", **kwargs
    ):
        # Same as run_with_hooks, but without no_grad for training use-cases.
        with self.hooks(list_of_hooks) as hooked_model:
            output = hooked_model(*args, **kwargs)
        if return_type == "output":
            return output
        if return_type == "loss":
            return self.contrastive_loss(
                output.logits_per_image, output.logits_per_text
            )
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def contrastive_loss(
        self,
        logits_per_image: Float[Tensor, "n_images n_prompts"],  # noqa: F722
        logits_per_text: Float[Tensor, "n_prompts n_images"],  # noqa: F722
    ):  # Assumes square matrices
        # Standard CLIP contrastive objective over image-text similarity matrix.
        assert logits_per_image.size()[0] == logits_per_image.size()[1], (
            "The number of prompts does not match the number of images."
        )
        batch_size = logits_per_image.size()[0]
        labels = torch.arange(batch_size).long().to(logits_per_image.device)
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        total_loss = (image_loss + text_loss) / 2
        return total_loss

    @contextmanager
    def hooks(self, hooks: List[Hook]):
        """

        This is a context manager for running a model with hooks. The funciton adds
        forward hooks to the model, and then returns the hooked model to be run with
        a foward pass. The funciton then cleans up by removing any hooks.

        Args:

          model VisionTransformer: The ViT that you want to run with the forward hook

          hooks List[Tuple[str, Callable]]: A list of forward hooks to add to the model.
            Each hook is a tuple of the module name, and the hook funciton.

        """
        hook_handles = []
        try:
            for hook in hooks:
                # Create a full hook funciton, with all the argumnets needed to run nn.module.register_forward_hook().
                # The hook functions are added to the output of the module.
                module = hook.get_module(self.model)
                handle = module.register_forward_hook(hook.function)
                hook_handles.append(handle)
            # Return control to caller while hooks are active.
            yield self.model
        finally:
            # Always clean up hooks, even when an exception is raised.
            # Without this, hooks would accumulate and corrupt later runs.
            for handle in hook_handles:
                handle.remove()

    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, *args, return_type="output", **kwargs):
        return self.forward(*args, return_type=return_type, **kwargs)

    def forward(self, *args, return_type="output", **kwargs):
        # Thin wrapper so callers can ask for model output or computed loss.
        if return_type == "output":
            return self.model(*args, **kwargs)
        elif return_type == "loss":
            output = self.model(*args, **kwargs)
            return self.contrastive_loss(
                output.logits_per_image, output.logits_per_text
            )
        else:
            raise Exception(
                f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'."
            )

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
