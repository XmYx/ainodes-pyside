# aiNodes - Stable Diffusion Desktop

<p align="left">
<a href="https://github.com/XmYx/ainodes-pyside/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/blob/main/aiNodes_webAPI_colab_v0_0_2_public.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>  
</p>

Colab API Server added with limited functionality

Please join our Discord for further information: https://discord.gg/XDQm9pk5pd

https://www.patreon.com/ainodes

Welcome to aiNodes, a desktop GUI with Deforum Art, Outpaint, Upscalers, and many more.

To install, please make sure you have python 3.10 on your system, then download or git clone the installer repository from:

https://www.python.org/downloads/release/python-3101/

For now we are back to a install batch file

for a first or clean install please do:

open a conda prompt
run clean_install.bat   (this will delete an existing ai-nodes env and rebuild a new one for rtx3xxx cards xformers should just work, rtx2xxx might run into issues)
clean_install will install everything and then start aiNodes

any further start can be made by start.bat it will just fire up the app and do not do any updates.

F.A.Q:

1. I have the following error: RuntimeError: failed to find interpreter for Builtin discover of python_spec='python3.10'
Solution: Please make sure you have Python version 3.10 installed on your system and added to your PATH, or in your conda environment you are running the launcher from.
https://realpython.com/add-python-to-path/
2. I have an access denied error when downloading the files from huggingface to the user\.cache folder
Solution can be to rename the existing folder to allow creating a new one, delete the existing one is kind the same but more agressive
also you can use HF_DATASETS_CACHE="/path/to/another/directory" as environment variable to point to a different folder

