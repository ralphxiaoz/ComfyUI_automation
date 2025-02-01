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
    update_node_input,
    update_vae_input,
    set_node_value
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
    # Load the workflow from a JSON file
    with open(get_path('workflow', 'Randomizer_controlNet.json'), 'r') as file:
        workflow = json.load(file)

    # Load CSV files
    with open(get_path('res', 'art_styles.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        art_styles = [row for row in reader if row.get('included', '').lower() == 'y']

    with open(get_path('res', 'models.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # Read all rows into memory first
        
        # Then get checkpoints
        # pool of checkpoints - for now, SD 1.5 & local only
        checkpoints = [row['Name'] for row in rows 
                        if row['Type'] == 'Checkpoint' 
                        and row['Base'] == "SD 1.5" 
                        and row['Location'] != 'External']

    # set checkpoint and loras =================================================================================================
    ckpt = "meinamix_v12Final.safetensors"
    set_vae = False
    vae_name = "vaeFtMse840000EmaPruned_vaeFtMse840k.safetensors"
    
    fixed_loras = []
    lora_categories = {
        "style": 0,
        "character_outfit": 0,
        "lighting": 0,
        "detail": 1,
        "crispness": 1,
        "quality": 1,
        "DOF": 1
    }

    embeddings = []

    use_random_seed = True
    
    """
    Common resolutions:
    - 16:9: 3840x2160, 1920x1080, 1280x768, 960x540, 912x512
    - 4:3: 1024x768, 800x600
    - 3:2: 1366x1024, 1280x800, 1152x768, 1024x682, 768x512
    """
    width = 912
    height = 512
    upscale_ratio = 2
    run_with_upscale = False
    sleep_time = 100

    use_art_style = False

    object_type = 14 # setting this to "target" will error out, need to fix

    randome_override = True

    for j in range (1, 500):
        
        # random override
        if randome_override:
            ckpt = random.choice(checkpoints) # this means no need to factor in random check points in get_model_params ***
            fixed_loras = []
            lora_categories = {
                "style": random.randint(0, 1),
                "lighting": random.randint(0, 1),
                "detail": random.randint(0, 1),
                "crispness": random.randint(0, 1),
                "quality": random.randint(0, 1)
            }
        object_type = random.randint(18, 19)
        
        # ControlNet setup
        if object_type == 13:
            input_img_name = random.choice(["snake_year_1.jpg"])
        elif object_type == 14:
            input_img_name = random.choice(["snake_year_2.jpg", "snake_year_3.jpg"])
        elif object_type == 15:
            input_img_name = random.choice(["snake_year_4.jpg", "snake_year_5.jpg"])
        elif object_type == 16:
            input_img_name = random.choice(["snake_year_6.jpg", "snake_year_7.jpg"])
        elif object_type == 17:
            input_img_name = random.choice(["snake_year_8.jpg", "snake_year_9.jpg"])
        elif object_type == 18:
            input_img_name = random.choice(["snake_year_10.jpg", "snake_year_11.jpg", "snake_year_12.jpg"])
        elif object_type == 19:
            input_img_name = random.choice(["snake_year_13.jpg", "snake_year_14.jpg"])

        if get_node_ID(workflow, "net1") is not None:
            logger.info(f"run.main: Setting input image to {input_img_name}")
            set_node_value(workflow, "Load Image", "image", input_img_name)

            set_node_value(workflow, "\ud83d\udd79\ufe0f CR Multi-ControlNet Stack", "switch_1", "On")
            set_node_value(workflow, "\ud83d\udd79\ufe0f CR Multi-ControlNet Stack", "switch_2", "On")

        logger.info(f"===== run.main: Running iteration {j} =====")
        style_name = random.choice(art_styles)['name'] if use_art_style else None
        logger.info(f"run.main: Style name: {style_name}")

        for i in range(1, 2):
            print("===== queueing workflow =====")
            seed = random.randint(1, 1000000000) if use_random_seed else 999999999
            loras = assemble_loras(ckpt, fixed_loras, lora_categories)
            logger.info(f"run.main: Selected loras: {loras}")
            num_loras = len(loras)
            models = get_model_params(num_loras=num_loras, checkpoint=ckpt, loras=loras, embeddings=embeddings)
            load_models_into_workflow(workflow, models)
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")

            checkpoint_used = models['checkpoint']
            loras_used = [models[f'lora{i}'] for i in range(1, num_loras + 1)]
            embeddings_used = ', '.join([models[f'embedding{i}'] for i in range(1, models.get('num_embeddings', 0) + 1)])
            
            # set the node values
            set_node_value(workflow, "CLIP Set Last Layer", "stop_at_clip_layer", -2)

            # set the KSampler node values
            set_KSampler(workflow, nodeTitle="KSampler", seed=seed, steps=30, cfg=8, sampler_name='dpmpp_2m', scheduler='karras', denoise=1)
            set_KSampler(workflow, nodeTitle="KS_up", seed=seed, steps=20, cfg=9, sampler_name='dpmpp_2m', scheduler='karras', denoise=0.4)
            set_positive_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, embeddings=embeddings_used, object_type=object_type, style_name=style_name)
            set_negative_prompt(workflow, ckpt_name=checkpoint_used, lora_names=loras_used, embeddings=embeddings_used, object_type=object_type, style_name=style_name)
            
            set_resolution(workflow, "Empty Latent Image", width, height)
            set_resolution(workflow, "Up_res", width*upscale_ratio, height*upscale_ratio)
            
            # run with upscale, set the input of save image to VAE Decode_scaled
            if run_with_upscale:
                update_node_input(workflow, "Save Image", "images", "VAE Decode_scaled")
            
            if set_vae:
                update_vae_input(workflow, vae_name)

            lora_prefixes = '-'.join([lora.replace(',', '_')[:5] for lora in loras_used])
            workflow["12"]["inputs"]["filename_prefix"] = f"{checkpoint_used.replace('.safetensors', '')}-{style_name}-{lora_prefixes}"
            
            # save the workflow to a file
            filename = f"last_execution_workflow.json"
            with open(get_path('workflow', filename), 'w') as outfile:
                json.dump(workflow, outfile, indent=4)
                outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')
            
            queue_workflow(workflow)
            
            if i % 1 == 0:
                time.sleep(sleep_time if run_with_upscale else 20)
    
    # Save the updated workflow
    with open(get_path('workflow', 'randomizer_updated.json'), 'w') as outfile:
        json.dump(workflow, outfile, indent=4)
        outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')