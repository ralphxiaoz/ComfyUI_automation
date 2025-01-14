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
    set_negative_prompt,
    update_node_input
)
from gen_prompt import gen_positive_prompt, gen_negative_prompt
from load_models import pick_models, load_models_into_workflow, queue_workflow
from config import get_path
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting workflow generation")
    # Load the workflow from a JSON file
    with open(get_path('workflow', 'Randomizer.json'), 'r') as file:
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

    # Load CSV files
    with open(get_path('res', 'models.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        checkpoints = [row['Name'] for row in reader if row['Type'] == 'Checkpoint']

    with open(get_path('res', 'art_styles.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        art_styles = [row for row in reader]

    ckpt = "meinamix_v12Final.safetensors"
    loras = []

    """
    Common resolutions:
    - 16:9: 3840x2160, 1920x1080, 1280x768, 960x540, 912x512
    - 4:3: 1024x768, 800x600
    - 3:2: 1280x800, 1152x768, 1024x682, 768x512
    """
    width = 512
    height = 768
    upscale_ratio = 2
    run_with_upscale = True

    for j in range (1, 500):
        style = random.choice(art_styles)
    #for style in art_styles:
        logger.info(f"Processing style: {style['name']}")
        style_name = style['name']
        for i in range(1, 10):
            print("===== queueing workflow =====")
            num_loras = random.randint(1, 4)
            models = pick_models(num_loras, checkpoint=ckpt, loras=loras)
            load_models_into_workflow(workflow, models)
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            checkpoint_used = models['checkpoint']
            loras_used = ', '.join([models[f'lora{i}'] for i in range(1, num_loras + 1)])
            
            set_KSampler(workflow, "KSampler", seed=random.randint(1, 1000000), steps=20, cfg=7, sampler_name='dpmpp_2m', scheduler='karras', denoise=1)
            set_positive_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, style_name=style_name)
            set_negative_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, style_name=style_name)
            
            set_resolution(workflow, "Empty Latent Image", width, height)
            set_resolution(workflow, "Up_res", width*upscale_ratio, height*upscale_ratio)
            
            # run with upscale, set the input of save image to VAE Decode_scaled
            if run_with_upscale:
                update_node_input(workflow, "Save Image", "images", "VAE Decode_scaled")
            
            workflow["12"]["inputs"]["filename_prefix"] = f"randomizer-{style_name}-{current_time}-{checkpoint_used.replace('.safetensors', '')}-{loras_used.replace('.safetensors', '')}"
            queue_workflow(workflow)
            print("===== workflow queued =====")
            if i % 9 == 0:
                time.sleep(60)
    
    # Save the updated workflow
    with open(get_path('workflow', 'randomizer_updated.json'), 'w') as outfile:
        json.dump(workflow, outfile, indent=4)
        outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')