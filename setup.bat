conda env create -f environment-installer.yaml
call activate ai-pixel
python -m pip install -e .
call deactivate