So facebook did give us something to play
This is no reference to the people who provided this in a way trying to gain benefit from naming them, Its just so they did it and its in the GIT url too.
We are not related to those guys in any way and we are not speaking for them or in their name.
We just point out they made it.

they call it xformers and what it does is, it speeds up calculation on your Nvidia GPU

if you got a 3xxx card,

just go here: https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/torch13/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl
download that file and pip install from root of the application


it managed to double the speed on a RTX2060

So here you go you want that beast and go render at light speed to. 


Well it is not that easy to begin with.

you must install Nvidia Cuda 11.3 later versions are simply not tested, feel free to test and please report back to us

https://developer.nvidia.com/cuda-11-3-1-download-archive

https://developer.nvidia.com/cuda-11.3.0-download-archive


You must install Visual Studio 2019 or 2022.

With 2022 you need to enforce the usage of newer compilers

As CUDA 11.3 is rather old, you need to force enable it to be built on MS Build Tools 2022. Do $env:NVCC_FLAGS = "-allow-unsupported-compiler" if on powershell, or set NVCC_FLAGS=-allow-unsupported-compiler if on cmd

so once you are set with the minimum basics you could go for a speedup by using Ninja for compiling

1. download ninja-win.zip from https://github.com/ninja-build/ninja/releases and unzip
2. place ninja.exe under C:\Windows OR add the full path to the extracted ninja.exe into system PATH
3. Run ninja -h in cmd and verify if you see a help message printed
4. Run the follow commands to start building. It should automatically use Ninja, no extra config is needed. You should see significantly higher CPU usage (40%+).

so now we have prepared: 

1. check Nvidia Cuda 11.3 is setup on your machine
2. check VS2019 or 2022. with 2022 you set the ENV var.
3. check You decided to use Ninja or dont its up to you.

Open a conda CMD

conda create --name xformers python=3.10
y
conda activate xformers
conda install -c pytorch -c conda-forge cudatoolkit=11.6 pytorch=1.12.1
y

git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .


at this point it should have compiled the xformers library.
It should have taken a considerable amount of time.
If it was just seconds it might not have worked. 
Don't trust iot telling you install success if it was just seconds

Now you have installed xformers into the env xformers. 
You can now start to run the environment_xformers.yaml into that environment

Or you can :

1. In xformers directory, navigate to the dist folder and copy the .whl file to the base directory of ai-pixels

2. In ai-pixels directory, install the .whl, change the name of the file in the command below if the name is different:

3. ./venv/scripts/activate
pip install xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl


After that you could start the App and set xformers to be used.

Try rendering and see if the speed is more than before
