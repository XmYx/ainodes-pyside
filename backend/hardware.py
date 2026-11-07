"""Hardware detection and memory planning for NVIDIA (CUDA), AMD (ROCm),
Intel (XPU), Apple (MPS) and CPU.

Everything here asks the driver for the *current* free memory instead of
trusting the card's nominal size, so other programs on the GPU (a running
ComfyUI, a game, the desktop) are taken into account. plan_memory() turns
that headroom into a loading strategy for a model of a given size.
"""
import gc
import os
import platform
from dataclasses import dataclass, field

GB = 1024 ** 3

# Keep this much free for activations, the VAE decode and other programs.
RESERVE_FRACTION = 0.10
RESERVE_MIN = int(1.0 * GB)


@dataclass
class Accelerator:
    backend: str  # cuda | rocm | xpu | mps | cpu
    index: int = 0
    name: str = "CPU"
    total_bytes: int = 0
    unified_memory: bool = False  # MPS and CPU share system RAM
    capability: str = ""  # "8.9" on CUDA, "gfx942" on ROCm
    bf16: bool = False
    fp8_compute: bool = False  # hardware fp8 matmul (fp8_e4m3fn_fast)
    fp8_storage: bool = False  # fp8 weights upcast per layer; saves memory everywhere it is allowed
    compile: bool = False

    @property
    def device(self):
        if self.backend in ("cuda", "rocm"):
            return f"cuda:{self.index}"  # ROCm builds of torch use the cuda namespace
        if self.backend == "xpu":
            return f"xpu:{self.index}"
        return self.backend

    @property
    def is_gpu(self):
        return self.backend != "cpu"


@dataclass
class HardwareInfo:
    accelerators: list = field(default_factory=list)
    ram_total: int = 0
    os: str = platform.system()

    @property
    def primary(self):
        return self.accelerators[0]


def _torch():
    import torch
    return torch


def _triton_available():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def _system_ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except Exception:
        return 0, 0


def _cuda_accelerators(torch):
    found = []
    rocm = getattr(torch.version, "hip", None) is not None
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if rocm:
            arch = getattr(props, "gcnArchName", "") or ""
            cap = arch.split(":")[0]
            fp8 = cap.startswith(("gfx942", "gfx950", "gfx12"))  # MI300/MI350, RDNA4
            bf16 = True
        else:
            major, minor = props.major, props.minor
            cap = f"{major}.{minor}"
            fp8 = (major, minor) >= (8, 9)  # Ada, Hopper, Blackwell
            bf16 = major >= 8
        found.append(Accelerator(
            backend="rocm" if rocm else "cuda", index=i, name=props.name, total_bytes=props.total_memory,
            capability=cap, bf16=bf16, fp8_compute=fp8, fp8_storage=hasattr(torch, "float8_e4m3fn"),
            compile=_triton_available() and platform.system() != "Darwin"))
    return found


def _xpu_accelerators(torch):
    found = []
    for i in range(torch.xpu.device_count()):
        props = torch.xpu.get_device_properties(i)
        found.append(Accelerator(
            backend="xpu", index=i, name=props.name, total_bytes=props.total_memory,
            bf16=True, fp8_compute=False, fp8_storage=hasattr(torch, "float8_e4m3fn"),
            compile=_triton_available()))
    return found


def _mps_accelerator(torch):
    total, _ = _system_ram()
    try:
        total = int(torch.mps.recommended_max_memory()) or total
    except Exception:
        pass
    mac_major = int((platform.mac_ver()[0] or "0").split(".")[0])
    return Accelerator(backend="mps", name=f"Apple GPU ({platform.machine()})", total_bytes=total,
                       unified_memory=True, bf16=mac_major >= 14,
                       fp8_compute=False, fp8_storage=False,  # MPS has no float8 tensors
                       compile=False)


def detect(force_device=None):
    """Find usable accelerators, best first. AINODES_DEVICE (cuda, xpu, mps, cpu) forces one."""
    force_device = (force_device or os.environ.get("AINODES_DEVICE", "")).lower()
    ram_total, _ = _system_ram()
    info = HardwareInfo(ram_total=ram_total)
    try:
        torch = _torch()
    except Exception:
        info.accelerators = [Accelerator(backend="cpu", total_bytes=ram_total, unified_memory=True)]
        return info
    candidates = []
    if torch.cuda.is_available():
        candidates += _cuda_accelerators(torch)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        candidates += _xpu_accelerators(torch)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        candidates.append(_mps_accelerator(torch))
    candidates.sort(key=lambda a: a.total_bytes, reverse=True)
    candidates.append(Accelerator(backend="cpu", total_bytes=ram_total, unified_memory=True, bf16=True,
                                  fp8_storage=hasattr(torch, "float8_e4m3fn")))
    if force_device:
        wanted = "cuda" if force_device.startswith("cuda") else force_device
        forced = [a for a in candidates if a.backend == wanted or (wanted == "cuda" and a.backend == "rocm")]
        candidates = forced + [a for a in candidates if a not in forced]
    info.accelerators = candidates
    return info


def memory_info(acc):
    """(free, total) bytes right now."""
    torch = _torch()
    if acc.backend in ("cuda", "rocm"):
        free, total = torch.cuda.mem_get_info(acc.index)
        return int(free), int(total)
    if acc.backend == "xpu":
        if hasattr(torch.xpu, "mem_get_info"):
            free, total = torch.xpu.mem_get_info(acc.index)
            return int(free), int(total)
        total = acc.total_bytes
        return int(total - torch.xpu.memory_reserved(acc.index)), int(total)
    _, available = _system_ram()
    if acc.backend == "mps":
        used = 0
        try:
            used = int(torch.mps.driver_allocated_memory())
        except Exception:
            pass
        return int(min(acc.total_bytes - used, available + used)), int(acc.total_bytes)
    return int(available), int(acc.total_bytes)


def headroom(acc, reserve_fraction=RESERVE_FRACTION, reserve_min=RESERVE_MIN):
    """Bytes a model may use now, after keeping a safety reserve."""
    free, total = memory_info(acc)
    reserve = max(reserve_min, int(total * reserve_fraction))
    return max(0, free - reserve)


def compute_dtype(acc):
    torch = _torch()
    if acc.backend == "cpu":
        return torch.bfloat16 if acc.bf16 else torch.float32
    return torch.bfloat16 if acc.bf16 else torch.float16


@dataclass
class MemoryPlan:
    strategy: str  # gpu | model_offload | group_offload | sequential_offload
    fp8_storage: bool
    reason: str


def plan_memory(acc, model_bytes, working_bytes=0, allow_fp8=True, available=None):
    """Pick how to load a model of `model_bytes` (weights at bf16/fp16) on `acc`.

    model_offload keeps one component (text encoder, transformer, VAE) on the
    GPU at a time; group_offload streams blocks of the transformer;
    sequential_offload streams layer by layer (slowest, smallest).
    """
    available = headroom(acc) if available is None else available
    need = model_bytes + working_bytes
    if acc.backend == "cpu":
        return MemoryPlan("gpu", False, "running on CPU")
    if need <= available:
        return MemoryPlan("gpu", False, f"fits: {need / GB:.1f} GB of {available / GB:.1f} GB free")
    if allow_fp8 and acc.fp8_storage and model_bytes // 2 + working_bytes <= available:
        return MemoryPlan("gpu", True, "fits with fp8 weights")
    largest_part = int(model_bytes * 0.6)  # the transformer / unet dominates
    if largest_part + working_bytes <= available:
        return MemoryPlan("model_offload", False, "components swap between GPU and RAM")
    if allow_fp8 and acc.fp8_storage and largest_part // 2 + working_bytes <= available:
        return MemoryPlan("model_offload", True, "components swap, fp8 weights")
    if available >= 2 * GB:
        return MemoryPlan("group_offload", acc.fp8_storage and allow_fp8, "transformer blocks are streamed")
    return MemoryPlan("sequential_offload", acc.fp8_storage and allow_fp8, "very little memory free")


def empty_cache():
    gc.collect()
    try:
        torch = _torch()
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if getattr(torch.version, "hip", None) is None:
            torch.cuda.ipc_collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def describe(info=None):
    """One line per accelerator for the UI."""
    info = info or detect()
    lines = []
    for acc in info.accelerators:
        if acc.backend == "cpu" and len(info.accelerators) > 1:
            continue
        try:
            free, total = memory_info(acc)
            mem = f"{free / GB:.1f} / {total / GB:.1f} GB free"
        except Exception:
            mem = f"{acc.total_bytes / GB:.1f} GB"
        features = [f for f, on in (("bf16", acc.bf16), ("fp8", acc.fp8_compute),
                                    ("fp8 weights", acc.fp8_storage and not acc.fp8_compute),
                                    ("torch.compile", acc.compile)) if on]
        cap = f" {acc.capability}" if acc.capability else ""
        lines.append(f"{acc.name} [{acc.backend}{cap}] {mem}" + (f" · {', '.join(features)}" if features else ""))
    return lines
