import importlib
import inspect
import re
import sys

import pytest


def test_lightning_rank_zero_only_old_import_path():
    import backend.compat  # noqa: F401
    from pytorch_lightning.utilities.distributed import rank_zero_only
    assert callable(rank_zero_only)


def test_transformers_helpers_available_from_modeling_utils():
    import backend.compat  # noqa: F401
    from transformers.modeling_utils import (apply_chunking_to_forward, find_pruneable_heads_and_indices,
                                             prune_linear_layer)
    heads, index = find_pruneable_heads_and_indices({1}, n_heads=3, head_size=2, already_pruned_heads=set())
    assert heads == {1}
    assert index.tolist() == [0, 1, 4, 5]
    assert callable(apply_chunking_to_forward) and callable(prune_linear_layer)


def test_basicsr_imports_on_current_torchvision():
    import backend.compat  # noqa: F401
    pytest.importorskip("basicsr")
    from basicsr.data import degradations
    assert hasattr(degradations, "add_jpg_compression")


def test_paths_registers_src_repos_when_present():
    # The repos are optional (tools/fetch_src_repos.py), but anything found must be on sys.path.
    import backend.paths as paths
    for directory in paths.paths.values():
        assert directory in sys.path


def test_src_repo_pins_are_full_commit_hashes():
    from tools.fetch_src_repos import REPOS
    assert len({name for name, _, _ in REPOS}) == len(REPOS)
    for name, url, commit in REPOS:
        assert url.startswith("https://github.com/"), name
        assert re.fullmatch(r"[0-9a-f]{40}", commit), name


def test_method_processor_ui_funcs_accept_clicked_item():
    # MethodProcessorWidget.item_clicked calls <method>_ui_func(item) for every list entry.
    pytest.importorskip("PySide6")
    module = importlib.import_module("plugins.method_processor.method_processor")
    cls = module.MethodProcessorWidget
    funcs = [name for name in dir(cls) if name.endswith("_ui_func")]
    assert funcs
    for name in funcs:
        params = list(inspect.signature(getattr(cls, name)).parameters)
        assert len(params) >= 2, f"{name} cannot take the clicked item"


@pytest.mark.slow
@pytest.mark.parametrize("module", ["ldm.models.diffusion.ddpm", "ldm_v2.models.diffusion.ddpm",
                                    "backend.deforum.deforum_adapter"])
def test_model_stack_imports(module):
    import backend.paths  # noqa: F401
    from backend.settings import load_default
    load_default()  # some modules read gs.system at import time, as in the app
    importlib.import_module(module)
