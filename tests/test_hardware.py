"""Hardware detection across vendors, simulated with a fake torch module."""
from types import SimpleNamespace

import pytest

from backend import hardware

GB = hardware.GB


class FakeCuda:
    def __init__(self, devices, free):
        self.devices, self.free = devices, free

    def is_available(self):
        return bool(self.devices)

    def device_count(self):
        return len(self.devices)

    def get_device_properties(self, i):
        return self.devices[i]

    def mem_get_info(self, i):
        return self.free[i], self.devices[i].total_memory

    def empty_cache(self):
        pass

    def ipc_collect(self):
        pass


def fake_torch(cuda=(), cuda_free=(), hip=None, xpu=(), xpu_free=(), mps=False, mps_total=0, mps_used=0):
    torch = SimpleNamespace()
    torch.float8_e4m3fn = object()
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
    torch.version = SimpleNamespace(hip=hip, cuda=None if hip else "12.8")
    torch.cuda = FakeCuda(list(cuda), list(cuda_free))
    xpu_devices = list(xpu)
    torch.xpu = SimpleNamespace(
        is_available=lambda: bool(xpu_devices), device_count=lambda: len(xpu_devices),
        get_device_properties=lambda i: xpu_devices[i],
        mem_get_info=lambda i: (xpu_free[i], xpu_devices[i].total_memory), empty_cache=lambda: None)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    torch.mps = SimpleNamespace(recommended_max_memory=lambda: mps_total,
                                driver_allocated_memory=lambda: mps_used, empty_cache=lambda: None)
    return torch


@pytest.fixture
def use_torch(monkeypatch):
    def install(torch, ram=(64 * GB, 40 * GB), triton=True):
        monkeypatch.setattr(hardware, "_torch", lambda: torch)
        monkeypatch.setattr(hardware, "_system_ram", lambda: ram)
        monkeypatch.setattr(hardware, "_triton_available", lambda: triton)
        monkeypatch.delenv("AINODES_DEVICE", raising=False)
        return torch
    return install


def cuda_props(name, total_gb, major, minor):
    return SimpleNamespace(name=name, total_memory=int(total_gb * GB), major=major, minor=minor)


def test_nvidia_ada_has_fp8_and_compile(use_torch):
    use_torch(fake_torch(cuda=[cuda_props("RTX 4090", 24, 8, 9)], cuda_free=[20 * GB]))
    acc = hardware.detect().primary
    assert (acc.backend, acc.device, acc.capability) == ("cuda", "cuda:0", "8.9")
    assert acc.bf16 and acc.fp8_compute and acc.fp8_storage and acc.compile
    assert hardware.memory_info(acc) == (20 * GB, 24 * GB)


def test_nvidia_turing_has_no_bf16_or_fp8(use_torch):
    use_torch(fake_torch(cuda=[cuda_props("RTX 2080", 8, 7, 5)], cuda_free=[6 * GB]))
    acc = hardware.detect().primary
    assert not acc.bf16 and not acc.fp8_compute
    assert hardware.compute_dtype(acc) == "fp16"


def test_amd_rocm(use_torch):
    mi300 = SimpleNamespace(name="AMD Instinct MI300X", total_memory=192 * GB, gcnArchName="gfx942:sramecc+:xnack-")
    rdna3 = SimpleNamespace(name="Radeon RX 7900 XTX", total_memory=24 * GB, gcnArchName="gfx1100")
    use_torch(fake_torch(cuda=[rdna3, mi300], cuda_free=[20 * GB, 180 * GB], hip="7.1"))
    accs = hardware.detect().accelerators
    assert [a.backend for a in accs] == ["rocm", "rocm", "cpu"]
    assert accs[0].name.startswith("AMD Instinct") and accs[0].fp8_compute  # biggest card first
    assert accs[1].capability == "gfx1100" and not accs[1].fp8_compute
    assert accs[0].device == "cuda:1"  # ROCm torch uses the cuda namespace


def test_intel_xpu(use_torch):
    arc = SimpleNamespace(name="Intel Arc B580", total_memory=12 * GB)
    use_torch(fake_torch(xpu=[arc], xpu_free=[11 * GB]))
    acc = hardware.detect().primary
    assert (acc.backend, acc.device) == ("xpu", "xpu:0")
    assert acc.bf16 and not acc.fp8_compute and acc.fp8_storage
    assert hardware.memory_info(acc) == (11 * GB, 12 * GB)


def test_apple_mps_uses_unified_memory(use_torch, monkeypatch):
    monkeypatch.setattr(hardware.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64"))
    use_torch(fake_torch(mps=True, mps_total=48 * GB, mps_used=8 * GB), ram=(64 * GB, 30 * GB))
    acc = hardware.detect().primary
    assert acc.backend == "mps" and acc.unified_memory and acc.bf16
    assert not acc.fp8_storage and not acc.compile  # no float8 tensors / inductor on MPS
    free, total = hardware.memory_info(acc)
    assert total == 48 * GB and free == 38 * GB  # limited by what the OS has available too


def test_cpu_only_and_forced_device(use_torch, monkeypatch):
    use_torch(fake_torch())
    assert hardware.detect().primary.backend == "cpu"
    use_torch(fake_torch(cuda=[cuda_props("RTX 4090", 24, 8, 9)], cuda_free=[20 * GB]))
    monkeypatch.setenv("AINODES_DEVICE", "cpu")
    assert hardware.detect().primary.backend == "cpu"


def test_headroom_keeps_a_reserve(use_torch):
    use_torch(fake_torch(cuda=[cuda_props("RTX 4090", 24, 8, 9)], cuda_free=[20 * GB]))
    acc = hardware.detect().primary
    reserve = max(hardware.RESERVE_MIN, int(24 * GB * hardware.RESERVE_FRACTION))
    assert hardware.headroom(acc) == 20 * GB - reserve
    use_torch(fake_torch(cuda=[cuda_props("RTX 4090", 24, 8, 9)], cuda_free=[1 * GB]))
    assert hardware.headroom(hardware.detect().primary) == 0


@pytest.mark.parametrize("available_gb, model_gb, strategy, fp8", [
    (20, 12, "gpu", False),               # fits as is
    (20, 30, "gpu", True),                # fits once weights are fp8
    (12, 30, "model_offload", True),      # transformer alone fits in fp8
    (20, 60, "model_offload", True),      # transformer fits once fp8
    (20, 80, "group_offload", True),      # stream transformer blocks
    (1, 12, "sequential_offload", True),  # almost nothing free
])
def test_plan_memory(available_gb, model_gb, strategy, fp8):
    acc = hardware.Accelerator(backend="cuda", fp8_storage=True, fp8_compute=True, bf16=True)
    plan = hardware.plan_memory(acc, model_gb * GB, 1 * GB, available=available_gb * GB)
    assert (plan.strategy, plan.fp8_storage) == (strategy, fp8)


def test_plan_memory_without_fp8_support():
    mps = hardware.Accelerator(backend="mps", fp8_storage=False)
    plan = hardware.plan_memory(mps, 30 * GB, 1 * GB, available=20 * GB)
    assert plan.strategy == "model_offload" and not plan.fp8_storage


def test_describe_lists_features(use_torch):
    use_torch(fake_torch(cuda=[cuda_props("RTX 4090", 24, 8, 9)], cuda_free=[20 * GB]))
    line = hardware.describe()[0]
    assert "RTX 4090" in line and "cuda 8.9" in line and "fp8" in line and "20.0 / 24.0 GB free" in line


def test_launcher_picks_wheels_by_vendor(monkeypatch):
    import launcher
    monkeypatch.delenv("AINODES_TORCH_INDEX", raising=False)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    for vendors, index in [({"nvidia", "intel"}, launcher.TORCH_CUDA_INDEX), ({"amd"}, launcher.TORCH_ROCM_INDEX),
                           ({"intel"}, launcher.TORCH_XPU_INDEX), (set(), launcher.TORCH_CPU_INDEX)]:
        monkeypatch.setattr(launcher, "gpu_vendors", lambda v=vendors: v)
        assert launcher.torch_index() == index
    monkeypatch.setattr(launcher.platform, "system", lambda: "Windows")
    monkeypatch.setattr(launcher, "gpu_vendors", lambda: {"amd"})
    assert launcher.torch_index() == launcher.TORCH_CPU_INDEX  # no ROCm wheels on Windows
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    assert launcher.torch_index() == ""  # PyPI wheels with MPS
