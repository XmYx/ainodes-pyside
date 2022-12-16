import itertools
import re
from backend.singleton import singleton
gs = singleton




def multi_dim_loop(runner,params):
    ints_vals=['steps']
    raw_prompts = params.prompts.split('\n')
    regex = r"(.*?)(--.*)"
    for prompt in raw_prompts:
        matches = re.match(regex, prompt, re.MULTILINE)
        work_prompt = matches.groups()[0]
        prompt_dimensions = matches.groups()[1]
        prompt_args = prompt_dimensions.split('--')
        args_dict={}
        for arg in prompt_args:
            if arg != '':
                arg = arg.rstrip()
                arg_name, args_list = arg.split('=')
                args_list=args_list.split(',')
                args_dict[arg_name] = args_list
        print(args_dict)
        pairs = [[(k, v) for v in args_dict[k]] for k in args_dict]
        arg_combinations = list(itertools.product(*pairs))
        for set in arg_combinations:
            for arg in set:
                name = arg[0]
                value = arg[1]
                if name == 'aesthetic_weight': gs.aesthetic_weight = float(value)
                if name == 'T': gs.T = int(value)
                if name == 'selected_aesthetic_embedding': gs.selected_aesthetic_embedding = str(value)
                if name == 'slerp': gs.slerp = bool(value)
                if name == 'aesthetic_imgs_text': gs.aesthetic_imgs_text = str(value)
                if name == 'slerp_angle': gs.slerp_angle = float(value)
                if name == 'aesthetic_text_negative': gs.aesthetic_text_negative = str(value)
                if name == 'gradient_scale': gs.lr = float(value)

                if name in ints_vals:
                    value = int(value)
                params.__dict__[name] = value
            params.prompts = work_prompt
            print(params)
            runner(params)
