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
from load_models import (
    get_model_params,
    load_models_into_workflow,
    queue_workflow,
    assemble_loras
)
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
    with open(get_path('res', 'art_styles.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        art_styles = [row for row in reader]

    with open(get_path('res', 'models.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # Read all rows into memory first
        
        # Check for commas in ALL model names
        for row in rows:
            if ',' in row['Name']:
                logger.error(f"Invalid model name found: {row['Name']}. Names cannot contain commas.")
                raise ValueError("Model names cannot contain commas. Ending job.")
        
        # Then get checkpoints
        checkpoints = [row['Name'] for row in rows if row['Type'] == 'Checkpoint']

    # set checkpoint and loras ===============================================================================================
    ckpt = "dreamshaper_8.safetensors"
    fixed_loras = []
    lora_categories = {
        "style": 0,
        "outfit": 0,
        "lighting": 0
    }
    loras = assemble_loras(ckpt, fixed_loras, lora_categories)
    logger.info(f"******Loras:******* {loras}")

    ckpt_base = None
    lora_bases = []

    # Find the base of the checkpoint
    for row in rows:
        if row['Type'] == 'Checkpoint' and row['Name'] == ckpt:
            ckpt_base = row['Base']
            break

    # Find the bases of the LoRAs
    for lora in loras:
        for row in rows:
            if row['Type'] == 'Lora' and row['Name'] == lora:
                lora_bases.append(row['Base'])
                break

    # in addition to the loras specified, add any loras that have the same base as the checkpoint
    # this may not be a good idea, hold off for now 
    # date: 2025-01-15
    # another way to implement this: create a function to assemble loras, categorize loras by types: quality, style, character, etc.
    # then, randomly select loras from each category - all quality, one or two style, one character, etc.
    available_loras = []

    # Find available LoRAs that have the same base as the checkpoint and are not in the specified loras
    for row in rows:
        if row['Type'] == 'Lora' and row['Base'] == ckpt_base and row['Name'] not in loras:
            available_loras.append(row['Name'])

    logger.info("Available LoRAs with the same base as the checkpoint: " + ", ".join(available_loras))

    # Check if the checkpoint and all LoRAs have the same base
    if ckpt_base and all(base == ckpt_base for base in lora_bases):
        logger.info("Checkpoint and LoRAs have the same base: " + ckpt_base)
    else:
        logger.warning("Checkpoint and LoRAs do not have the same base. Checkpoint base: " + str(ckpt_base) + ", LoRA bases: " + str(lora_bases))

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
    #for style in art_styles:

        #style_name = style['name']
        style_name = "Photography - Fashion"

        for i in range(1, 5):
            print("===== queueing workflow =====")
            num_loras = random.randint(1, 5) if not loras else len(loras)
            models = get_model_params(num_loras, checkpoint=ckpt, loras=loras)
            load_models_into_workflow(workflow, models)
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")

            checkpoint_used = models['checkpoint']
            loras_used = ', '.join([models[f'lora{i}'] for i in range(1, num_loras + 1)])
            embeddings_used = ', '.join([models[f'embedding{i}'] for i in range(1, models.get('num_embeddings', 0) + 1)])
            
            set_KSampler(workflow, "KSampler", seed=random.randint(1, 1000000), steps=20, cfg=7, sampler_name='dpmpp_2m', scheduler='karras', denoise=1)
            set_positive_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, embeddings=embeddings_used, style_name=style_name)
            set_negative_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, embeddings=embeddings_used, style_name=style_name)
            
            set_resolution(workflow, "Empty Latent Image", width, height)
            set_resolution(workflow, "Up_res", width*upscale_ratio, height*upscale_ratio)
            
            # run with upscale, set the input of save image to VAE Decode_scaled
            if run_with_upscale:
                update_node_input(workflow, "Save Image", "images", "VAE Decode_scaled")
            
            lora_prefixes = '-'.join([lora[:5] for lora in loras_used.split(', ')])
            workflow["12"]["inputs"]["filename_prefix"] = f"{checkpoint_used.replace('.safetensors', '')}-{style_name}-{lora_prefixes}"
            
            # save the workflow to a file
            filename = f"last_execution_workflow.json"
            with open(get_path('workflow', filename), 'w') as outfile:
                json.dump(workflow, outfile, indent=4)
                outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')
            
            queue_workflow(workflow)
            print("===== workflow queued =====")
            
            if i % 1 == 0:
                time.sleep(30 if run_with_upscale else 5)
    
    # Save the updated workflow
    with open(get_path('workflow', 'randomizer_updated.json'), 'w') as outfile:
        json.dump(workflow, outfile, indent=4)
        outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')