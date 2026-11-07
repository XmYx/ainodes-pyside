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


def _patch_clip_text_model():
    # transformers 5 flattened CLIPTextModel: embeddings / encoder / final_layer_norm
    # are now direct children instead of living under .text_model. The ldm code
    # still reaches through .text_model, and every SD 1.x checkpoint stores the
    # text encoder as "...transformer.text_model.*".
    try:
        from transformers import CLIPTextModel
    except ImportError:
        return
    probe = getattr(CLIPTextModel, "_ainodes_flat_patch", None)
    if probe is not None or "text_model" in getattr(CLIPTextModel, "__annotations__", {}):
        return
    import inspect
    if "self.text_model" in inspect.getsource(CLIPTextModel.__init__):
        return  # old layout, nothing to do

    original_forward = CLIPTextModel.forward

    class TextModelView:
        """Stand-in for the old inner CLIPTextTransformer.

        Attribute reads go to the model, but `forward` is the view's own, so code
        that replaces transformer.text_model.forward and transformer.forward
        separately (the ldm embedding managers do) keeps two distinct levels.
        """

        def __init__(self, model):
            object.__setattr__(self, "_model", model)
            object.__setattr__(self, "forward", original_forward.__get__(model))

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_model"), name)

        def __setattr__(self, name, value):
            if name == "forward":
                object.__setattr__(self, name, value)
            else:
                setattr(object.__getattribute__(self, "_model"), name, value)

        def __call__(self, *args, **kwargs):
            return object.__getattribute__(self, "forward")(*args, **kwargs)

    def text_model(self):
        view = self.__dict__.get("_ainodes_text_model_view")
        if view is None:
            view = TextModelView(self)
            self.__dict__["_ainodes_text_model_view"] = view
        return view

    CLIPTextModel.text_model = property(text_model)

    def strip_text_model(state_dict, prefix, *args):
        old = prefix + "text_model."
        for key in [k for k in state_dict if k.startswith(old)]:
            state_dict[prefix + key[len(old):]] = state_dict.pop(key)

    original_init = CLIPTextModel.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._register_load_state_dict_pre_hook(strip_text_model)

    CLIPTextModel.__init__ = __init__
    CLIPTextModel._ainodes_flat_patch = True


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
    _patch_clip_text_model()
    _patch_lightning()
    _patch_torchvision()


apply()
