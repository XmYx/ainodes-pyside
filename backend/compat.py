"""Shims that let the vendored 2022/2023 model code run on current libraries.

The bundled ldm code and the repos cloned into ./src (BLIP, k-diffusion,
taming, ...) import helpers that newer torch / transformers /
pytorch_lightning releases moved or removed. Instead of patching the
third-party clones, the old names are put back here. backend.paths
imports this module, so it runs before any model code is loaded.
"""
import sys
import types


def _patch_transformers():
    try:
        import torch
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


def _patch_lightning():
    # pytorch_lightning 1.x had rank_zero_only in utilities.distributed;
    # 2.x keeps it in utilities.rank_zero and dropped the old module.
    try:
        import pytorch_lightning.utilities.distributed  # noqa: F401
        return
    except ImportError:
        pass
    try:
        from pytorch_lightning.utilities import rank_zero
    except ImportError:
        return
    module = types.ModuleType("pytorch_lightning.utilities.distributed")
    module.rank_zero_only = rank_zero.rank_zero_only
    sys.modules[module.__name__] = module


def _patch_torchvision():
    # basicsr (GFPGAN / RealESRGAN) imports functional_tensor, removed in torchvision 0.17.
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ImportError:
        pass
    try:
        from torchvision.transforms import functional
    except ImportError:
        return
    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module.__name__] = module


def apply():
    _patch_transformers()
    _patch_lightning()
    _patch_torchvision()


apply()
