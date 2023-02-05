import random
import secrets

from PySide6.QtCore import QObject, Signal
from transformers import pipeline, set_seed

from backend import seeder
from backend.prompt_ai.nsp.nsp_pantry import parser

gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')


class Callbacks(QObject):
    ai_prompt_ready = Signal(str)
    status_update = Signal(str)


class AiPrompt:
    def __init__(self):
        self.signals = Callbacks()

    def get_prompts(self, prompts_txt='', progress_callback=None):
        out_text = ''
        prompts_array = prompts_txt.split('\n')

        for prompt in prompts_array:
            self.signals.status_update.emit(f'working on Prompt: {prompt}')
            out_text += self.generate_prompt(prompt)
        self.signals.ai_prompt_ready.emit(out_text)

    def generate_prompt(self, starting_text):
        response_end = ''
        all_prompts = []
        num_prompts = 0
        try:
            for count in range(2):
                seed = seeder.get_strong_seed(1000000)
                set_seed(seed)
                if starting_text == "":
                    nsp = parser()
                    nsp_keys = nsp.get_nsp_keys()

                    rand_number_of_topics = random.randrange(5)
                    text_array = []
                    starting_text = ' '
                    for rnt in range(rand_number_of_topics):
                        text_array.append(nsp.parse(nsp_keys[random.randrange(len(nsp_keys)-1)]))
                    starting_text = starting_text.join(text_array)
                response = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)
                for prompt in response:
                    temp_prompt = prompt['generated_text']
                    temp_prompt = temp_prompt.replace('\n','')
                    if temp_prompt not in all_prompts:
                        num_prompts += 1
                        all_prompts.append(temp_prompt)
                self.signals.status_update.emit(f'{num_prompts} prompts already dreamed, stay patient..')
        except Exception as e:
            print('AI Prompt failed: ', e)
            all_prompts = ''
        finally:
            return '\n\n'.join(all_prompts)
