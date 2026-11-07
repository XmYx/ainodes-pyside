"""Set up aiNodes in ./.venv (first run, or when requirements change) and start it.

Run through start.sh (Linux / macOS) or start.bat (Windows). Uses only the
standard library so any Python 3.10-3.12 (or uv) can run it.

    --install-only   set up the environment, do not start the app
    --reinstall      reinstall dependencies even if nothing changed
    --test           run the unit tests instead of the app
    --cpu            install the CPU build of torch
Any other arguments are passed to the app.
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
STAMP = os.path.join(VENV, ".ainodes-install")
PYTHON_VERSION = "3.12"
SUPPORTED = ((3, 10), (3, 12))
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_ROCM_INDEX = "https://download.pytorch.org/whl/rocm7.1"
TORCH_XPU_INDEX = "https://download.pytorch.org/whl/xpu"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def log(message):
    print(f"[aiNodes] {message}", flush=True)


def run(cmd, **kwargs):
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, **kwargs)


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def gpu_vendors():
    """PCI display vendors present: nvidia, amd, intel (best effort, no torch needed)."""
    vendors = set()
    if shutil.which("nvidia-smi"):
        vendors.add("nvidia")
    if shutil.which("rocminfo") or shutil.which("rocm-smi") or os.path.exists("/opt/rocm"):
        vendors.add("amd")
    ids = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
    drm = "/sys/class/drm"
    if os.path.isdir(drm):
        for card in os.listdir(drm):
            path = os.path.join(drm, card, "device", "vendor")
            if card.startswith("card") and "-" not in card and os.path.exists(path):
                vendors.add(ids.get(open(path).read().strip(), "other"))
    if os.name == "nt":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_VideoController).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.lower()
            for key, vendor in (("nvidia", "nvidia"), ("radeon", "amd"), ("amd", "amd"), ("arc", "intel")):
                if key in out:
                    vendors.add(vendor)
        except Exception:
            pass
    return vendors


def torch_index(force_cpu=False):
    """Wheel index for torch, by GPU: NVIDIA CUDA, AMD ROCm (Linux), Intel XPU, macOS PyPI (MPS)."""
    if os.environ.get("AINODES_TORCH_INDEX"):
        return os.environ["AINODES_TORCH_INDEX"]
    if platform.system() == "Darwin":
        return ""
    if force_cpu:
        return TORCH_CPU_INDEX
    vendors = gpu_vendors()
    if "nvidia" in vendors:
        return TORCH_CUDA_INDEX
    if "amd" in vendors and platform.system() == "Linux":
        return TORCH_ROCM_INDEX
    if "intel" in vendors:
        # integrated Intel graphics also match; XPU wheels fall back to CPU cleanly there
        return TORCH_XPU_INDEX
    if "amd" in vendors:
        log("AMD GPU on Windows: PyTorch has no ROCm wheels there, using CPU. "
            "Use the ComfyUI backend (ComfyUI-Zluda / ROCm) for GPU speed.")
    return TORCH_CPU_INDEX


def install_fingerprint(index):
    digest = hashlib.sha256(index.encode())
    with open(os.path.join(ROOT, "requirements.txt"), "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()


def create_venv(uv):
    if os.path.exists(venv_python()):
        return
    if uv:
        run([uv, "venv", "--python", PYTHON_VERSION, VENV])
        return
    if not SUPPORTED[0] <= sys.version_info[:2] <= SUPPORTED[1]:
        sys.exit(f"aiNodes needs Python 3.10-3.12 (this is {platform.python_version()}). "
                 "Install one of those, or install uv (https://docs.astral.sh/uv/) and it will fetch Python for you.")
    run([sys.executable, "-m", "venv", VENV])
    run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"])


def pip_install(uv, args):
    if uv:
        run([uv, "pip", "install", "--python", venv_python(), *args])
    else:
        run([venv_python(), "-m", "pip", "install", *args])


def install(uv, force=False, force_cpu=False):
    create_venv(uv)
    index = torch_index(force_cpu)
    fingerprint = install_fingerprint(index)
    current = open(STAMP).read().strip() if os.path.exists(STAMP) else ""
    if force or current != fingerprint:
        log("Installing dependencies, the first run takes a while")
        pip_install(uv, ["torch", "torchvision", *(["--index-url", index] if index else [])])
        pip_install(uv, ["-r", os.path.join(ROOT, "requirements.txt")])
        with open(STAMP, "w") as f:
            f.write(fingerprint)
    sys.path.insert(0, ROOT)
    from tools.fetch_src_repos import fetch
    fetch(os.path.join(ROOT, "src"))


def main(argv):
    flags = {"--install-only", "--reinstall", "--test", "--cpu"}
    options = {a for a in argv if a in flags}
    app_args = [a for a in argv if a not in flags]
    uv = shutil.which("uv")
    install(uv, force="--reinstall" in options, force_cpu="--cpu" in options)
    if "--install-only" in options:
        log("Environment ready")
        return 0
    if "--test" in options:
        return subprocess.call([venv_python(), "-m", "pytest", *app_args], cwd=ROOT)
    return subprocess.call([venv_python(), os.path.join(ROOT, "start.py"), *app_args], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
