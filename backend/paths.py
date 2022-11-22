import argparse
import os
import sys
#import backend.hypernetworks.modules.safe

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
models_path = os.path.join(script_path, "models")
sys.path.insert(0, script_path)

# search for directory of stable diffusion in following places
sd_path = os.path.join(script_path,'src')
#assert sd_path is not None, "Couldn't find Stable Diffusion in any of: " + str(possible_sd_paths)

path_dirs = [
    #(sd_path, 'ldm', 'Stable Diffusion', []),
    (os.path.join(sd_path, 'taming-transformers'), 'taming', 'Taming Transformers', []),
    (os.path.join(sd_path, 'CodeFormer'), 'inference_codeformer.py', 'CodeFormer', []),
    (os.path.join(sd_path, 'BLIP'), 'models/blip.py', 'BLIP', ["atstart"]),
    (os.path.join(sd_path, 'k-diffusion'), 'k_diffusion/sampling.py', 'k_diffusion', ["atstart"]),
    (os.path.join(sd_path, 'AdaBins'), 'model_io.py', 'AdaBins', []),
    (os.path.join(sd_path, 'MiDaS'), 'run.py', 'MiDaS', []),
    (os.path.join(sd_path, 'pytorch3d-lite'), 'py3d_tools.py', 'pytorch-lite', []),
    (os.path.join(sd_path, 'realesrgan'), 'inference_realesrgan.py', 'realesrgan', []),
    (os.path.join(sd_path, 'improved-aesthetic-predictor'), 'train_predictor.py', 'improved-aesthetic-predictor', []),
]
paths = {}

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        if d not in sys.path:
            if "atstart" in options:
                sys.path.insert(0, d)
            else:
                sys.path.append(d)
            paths[what] = d
