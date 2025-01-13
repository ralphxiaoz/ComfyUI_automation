import csv
import random
import json
from datetime import datetime
from urllib import request
import urllib.error
import time
from node_manipulation import (
    set_number_of_loras, 
    set_lora, 
    get_node_ID, 
    set_KSampler, 
    set_resolution,
    set_positive_prompt,
    set_negative_prompt
)
from gen_prompt import gen_positive_prompt, gen_negative_prompt
from load_models import pick_models
from load_models import load_models_into_workflow
from load_models import queue_workflow

if __name__ == "__main__":
    # Load the workflow from a JSON file
    with open('Randomizer.json', 'r') as file:
        workflow = json.load(file)
    
    """
    # Get models from pick_models
    num_loras = random.randint(1, 4)
    models = pick_models(num_loras, checkpoint="meinamix_v12Final.safetensors", loras=None)

    # Load models into the workflow
    load_models_into_workflow(workflow, models)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_used = models['checkpoint']
    loras_used = ', '.join([models[f'lora{i}'] for i in range(1, num_loras + 1)])

    print("Number of LoRAs:", num_loras)
    print("Loras used:", loras_used)

    # Set the filename for saving the image
    workflow["12"]["inputs"]["filename_prefix"] = f"randomizer-{current_time}-{checkpoint_used.replace('.safetensors', '')}-{loras_used.replace('.safetensors', '')}"
    """

    with open('models.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        checkpoints = [row['Name'] for row in reader if row['Type'] == 'Checkpoint']

    with open('art_styles.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        art_styles = [row for row in reader]

    ckpt = "dreamshaper_8.safetensors"

    for style in art_styles:
        style_name = style['name']
        for i in range(1, 3):
            print("===== queueing workflow =====")
            num_loras = random.randint(1, 4)
            models = pick_models(num_loras, checkpoint=ckpt, loras=None)
            load_models_into_workflow(workflow, models)
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            checkpoint_used = models['checkpoint']
            loras_used = ', '.join([models[f'lora{i}'] for i in range(1, num_loras + 1)])
            set_KSampler(workflow, "KSampler", seed=random.randint(1, 1000000), steps=20, cfg=7, sampler_name='dpmpp_2m', scheduler='karras', denoise=1)
            set_positive_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, style_name=style_name)
            set_negative_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, style_name=style_name)
            set_resolution(workflow, "Empty Latent Image", 768, 768)
            workflow["12"]["inputs"]["filename_prefix"] = f"randomizer-{style_name}-{current_time}-{checkpoint_used.replace('.safetensors', '')}-{loras_used.replace('.safetensors', '')}"
            queue_workflow(workflow)
            print("===== workflow queued =====")
            if i % 2 == 0:
                time.sleep(100)
    
    # Save the updated workflow to a new JSON file
    with open('randomizer_updated.json', 'w') as outfile:
        json.dump(workflow, outfile, indent=4)
        outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')  # Commented out datetime stamp