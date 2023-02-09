# aiNodes - Stable Diffusion Desktop

<p align="left">
<a href="https://github.com/XmYx/ainodes-pyside/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/XmYx/ainodes-pyside"></a>
<a href="https://github.com/XmYx/ainodes-pyside/blob/main/aiNodes_webAPI_colab_v0_0_2_public.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>  
</p>

Please join our Discord for further information: https://discord.gg/XDQm9pk5pd

Welcome to aiNodes, a desktop GUI with Deforum Art, Outpaint, Upscalers, and many more.

This product does not yet have a sophisticated installer mechanics.
So you have to follow the few but very simple installation steps

1. Install Conda, download from here https://docs.conda.io/en/latest/miniconda.html
2. Install GIT https://git-scm.com/download/win.
3. Once Conda is installed start the conda prompt from start menue. Use the Conda prompt not the Conda powershell.

![img.png](installImages/img_c.png)

![img.png](installImages/img.png)

4. from with in the conda prompt you follow thi steps:
5. clone your copy of aiNodes with git clone https://github.com/XmYx/ainodes-pyside
6. Cd to your copy of aiNondes

![img.png](installImages/img2.png)
7.	Run clean_install.bat as normal, non-administrator, user.

![img.png](installImages/img3.png)
8. This will run quite a while and do all the necessary steps to get the application installed and finally started for the first time.

![img.png](installImages/img4.png)

![img.png](installImages/img_running.png)

any further start can be made by start.bat it will just fire up the app and do not do any updates.

# F.A.Q:

1. I have the following error: RuntimeError: failed to find interpreter for Builtin discover of python_spec='python3.10'
Solution: Please make sure you have Python version 3.10 installed on your system and added to your PATH, or in your conda environment you are running the launcher from.
https://realpython.com/add-python-to-path/
2. I have access denied error when downloading the files from huggingface to the user\.cache folder
Solution can be to rename the existing folder to allow creating a new one, delete the existing one is kind the same but more agressive
also you can use HF_DATASETS_CACHE="/path/to/another/directory" as environment variable to point to a different folder

# Feature List
- Deforum as render Backend
- Deforum like prompt weighting
- text to image
- image to image
- text to video
- video to video
- Automatic1111 like prompting

- seamless for render textures and stuff
- HiRes Render two phase Hires Rendering
- multidimensional Prompting allows to prompt settings with lists of values which then get looped, produces massive amounts of images with different settings
- multi Model Batch allows your prompt(s) to be rendered with different models in one single run. Just prepare run and go get coffee
- textual inversion 
- VAE
- hypernetworks
- karras
- plotting with XY axis enter a list of values to be used for different settings on X and Y axis to in the end get a grid image showing the results for easy compare
- plugins A list of plugins exists like training and webcam fun list will grow over time, and you're welcome to add your own plugins or a Colab API Server with limited functionality
- canvas for manual, batched in and outpaint as well as a Batch Hires image to image process
- model downloads in UI, just search civit AI for a new model, download it and directly start using it, no need to copy files or anything just stay in the tool
- image lab with many features most of them are bulk centered like
  - image to text which generates a ascii image of your image (It's there mainly for fun)
  - image crop helps you to crop your images for training
  - upscale use realesrgan and gfpgan to upscale and make nicer faces
  - watermark add visible watermarks to your image to prevent theft
  - interrogation make images to prompts with interrogation or clip interrogation
  - aesthetic prediction help to sort your large amounts of images based on aesthetic guess from an AI
  - merge models 1.x or 2.x
  - upscale 2.0 it's a SD 2.0 feature which is so hungry for VRam that it nearly makes no sense but we added it anyways
- search for prompts at lexica or kreas getting not only prompts but also sample images
- as well as search for prompts only text
  - use interrogation to get prompts from your holiday photos
  - use an AI to help make better prompts

All this is hiding behind that very harmless looking UI once you installed it and run it for the first time.
Click on show / hide options to enable the full power of this UI to help you make more art in less time






