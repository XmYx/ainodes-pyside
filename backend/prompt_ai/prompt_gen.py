from transformers import pipeline, set_seed
import random, re, string
from pyparsing import line
from backend.prompt_ai.nsp.nsp_pantry import parser
import json

gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')


def generate_prompt(starting_text):
    print('was ai prompt')

    response_end = ''
    all_prompts = []
    try:
        for count in range(4):
            seed = random.randint(100, 1000000)
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

            print('response' + str(count))
            print(response)
            print(type(response))

            for prompt in response:
                temp_prompt = prompt['generated_text']
                temp_prompt = temp_prompt.replace('\n','')
                print(temp_prompt)
                if temp_prompt not in all_prompts:
                    all_prompts.append(temp_prompt)
    except Exception as e:
        print('AI Prompt failed: ', e)
    finally:
        return '\n\n'.join(all_prompts)
