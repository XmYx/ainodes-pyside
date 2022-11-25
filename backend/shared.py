from datetime import datetime
from backend.singleton import singleton

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
    if gs.models:
        if len(gs.models.keys()) > 0:
            models = gs.models.keys()
            for model in models:
                if model != keep:
                    del gs.models[model]
