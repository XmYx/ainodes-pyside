import subprocess
import os



def run_shivams_dreambooth():
    MODEL_NAME = "runwayml/stable-diffusion-v1-5" #@param {type:"string"}
    import json
    import os
    #@markdown Enter the directory name to save model at.

    OUTPUT_DIR = "" #@param {type:"string"}
    OUTPUT_DIR = "dreambooth_test" + OUTPUT_DIR

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Weights will be saved at {OUTPUT_DIR}")

    # You can also add multiple concepts here. Try tweaking `--max_train_steps` accordingly.

    concepts_list = [
        {
            "instance_prompt":      "photo of zwx dog",
            "class_prompt":         "photo of a dog",
            "instance_data_dir":    "/test",
            "class_data_dir":       "/test"
        },
    #     {
    #         "instance_prompt":      "photo of ukj person",
    #         "class_prompt":         "photo of a person",
    #         "instance_data_dir":    "/content/data/ukj",
    #         "class_data_dir":       "/content/data/person"
    #     }
    ]

    # `class_data_dir` contains regularization images
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    command = ['accelerate', 'launch', 'plugins/training/shivams_dreambooth/dreambooth.py',
               '--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5',
               '--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse',
               '--output_dir=$OUTPUT_DIR',
               '--revision=fp16',
               '--with_prior_preservation', '--prior_loss_weight=1.0',
               '--seed=1337',
               '--resolution=512',
               '--train_batch_size=1',
               '--train_text_encoder',
               '--mixed_precision=fp16',
               '--use_8bit_adam',
               '--gradient_accumulation_steps=1',
               '--learning_rate=1e-6',
               '--lr_scheduler=constant',
               '--lr_warmup_steps=0',
               '--num_class_images=50',
               '--sample_batch_size=4',
               '--max_train_steps=800',
               '--save_interval=10000',
               '--save_sample_prompt=photo of zwx dog',
               '--concepts_list=concepts_list.json']

    subprocess.run(command)

