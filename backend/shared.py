import os
from datetime import datetime

import urllib3

from backend.singleton import singleton
from backend.torch_gc import torch_gc
gs = singleton

def save_last_prompt(prompt_html, prompt_txt):
    f = open('configs/ainodes/last_prompt.txt', mode="w", encoding="utf-8")
    f.write(prompt_html)
    f.close()
    f = open('configs/ainodes/prompt_history.txt', mode="a", encoding="utf-8")
    dt = datetime.now()
    f.write(f'{dt}: {prompt_html}\n')
    f.close()
    f = open('configs/ainodes/prompt_history_raw.txt', mode="a", encoding="utf-8")
    f.write(f'{dt}: {prompt_txt}\n')
    f.close()


def model_killer(keep=''):
    temp = None
    if keep in gs.models:
        temp = gs.models[keep]
        gs.models = {keep: temp}
    else:
        gs.models = {}
    del temp
    torch_gc()


def check_support_model_exists(model_name, model_path, model_url):
    out_path = os.path.join(model_path, model_name)
    if os.path.isfile(out_path):
        return out_path
    else:
        os.makedirs(gs.system.support_models, exist_ok=True)
        print(f'Installing {model_name} Model to {model_path} from {model_url}')
        http = urllib3.PoolManager()
        r = http.request('GET', model_url, preload_content=False)
        chunk_size = 1024
        with open(out_path, 'wb') as out:
            while True:
                data = r.read(chunk_size)
                if not data:
                    break
                out.write(data)
        r.release_conn()

        return out_path
