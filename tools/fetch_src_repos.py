"""Clone the third-party source repos aiNodes imports from ./src at pinned commits.

launcher.py runs this on every start; it is a no-op once the clones are at
the pinned commits.
Run from the project root: python tools/fetch_src_repos.py
"""
import os
import subprocess
import sys

REPOS = [
    ("taming-transformers", "https://github.com/CompVis/taming-transformers.git", "24268930bf1dce879235a7fddd0b2355b84d7ea6"),
    ("k-diffusion", "https://github.com/crowsonkb/k-diffusion.git", "5b3af030dd83e0297272d861c19477735d0317ec"),
    ("CodeFormer", "https://github.com/sczhou/CodeFormer.git", "c5b4593074ba6214284d6acd5f1719b6c5d739af"),
    ("BLIP", "https://github.com/salesforce/BLIP.git", "48211a1594f1321b00f14c9f7a5b4813144b2fb9"),
    ("realesrgan", "https://github.com/xinntao/Real-ESRGAN.git", "64ad194ddaf9c4d8c4b0d1b98cac6d89d3ea0d11"),
    ("AdaBins", "https://github.com/osi1880vr/AdaBins.git", "4524615236f5f486381fac2f9c624f20dedf324f"),
    ("MiDaS", "https://github.com/isl-org/MiDaS.git", "66882994a432727317267145dc3c2e47ec78c38a"),
    ("pytorch3d-lite", "https://github.com/osi1880vr/pytorch3d-lite.git", "4070975c1d6e4de7c87848a53b603f6b29711e55"),
    ("improved-aesthetic-predictor", "https://github.com/christophschuhmann/improved-aesthetic-predictor.git", "fe88a163f4661b4ddabba0751ff645e2e620746e"),
]


def git(*args):
    subprocess.run(["git", *args], check=True)


def fetch(src_dir):
    os.makedirs(src_dir, exist_ok=True)
    for name, url, commit in REPOS:
        path = os.path.join(src_dir, name)
        if not os.path.exists(path):
            print(f"Cloning {name}")
            git("clone", "-q", url, path)
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != commit:
            git("-C", path, "-c", "advice.detachedHead=false", "checkout", "-q", commit)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fetch(sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "src"))
