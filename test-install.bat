call conda update -n base -c defaults conda -y
call conda activate base
call conda env remove -n ai-pixel
call conda env create -f environment-installer.yaml
call conda activate ai-pixel
setup.bat
