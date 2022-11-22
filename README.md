# aiNodes - Stable Diffusion Desktop

Please join our Discord for further information: https://discord.gg/XDQm9pk5pd

Welcome to our first alpha release, please expect many improvements in a very short amount of time.

prerequisities: A working conda install, MS Visual Studio 2019 with Windows 10 SDK, Cuda 11.6
https://developer.nvidia.com/cuda-11-6-0-download-archive
https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=community&rel=16&utm_medium=microsoft&utm_campaign=download+from+relnotes&utm_content=vs2019ga+button
https://www.anaconda.com/products/distribution

install/run:
conda env create -n ainodes -f environment_310.yaml
conda activate ainodes
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
git clone https://github.com/facebookresearch/xformers
cd xformers
(optionally) pip install ninja
git submodule update --init --recursive
pip install -r requirements-test.txt
pip install -e .

then running should be as simple as activating the environment with:
conda activate ainodes
python frontend/main_app.py

You can do both manually.

Linux, macOS installers coming up.
