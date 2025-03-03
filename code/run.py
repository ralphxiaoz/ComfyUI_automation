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
from gen_prompt import gen_positive_prompt, gen_negative_prompt, get_object
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
                        and row['Location'] != 'External'
                        and row['Excluded'] != 'y']


    # set checkpoint and loras =================================================================================================
    ckpt = "helloartOil_helloartOilV10kvae.safetensors"
    set_vae = False
    vae_name = "vaeFtMse840000EmaPruned_vaeFtMse840k.safetensors"
    
    fixed_loras = []
    lora_categories = {
        "style": 1,
        "character_outfit": 0,
        "lighting": 1,
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
    width = 512
    height = 512
    upscale_ratio = 2
    run_with_upscale = True
    sleep_time_up = 80
    sleep_time_regular = 30

    use_art_style = True

    object_type = "target" # setting this to "target" will error out, need to fix

    random_override = True

    for j in range (1, 500):
        
        # random override
        if random_override:
            ckpt = random.choice(checkpoints) # this means no need to factor in random check points in get_model_params ***
            fixed_loras = []

            lora_categories = {
                #"style": random.randint(0, 1),
                #"lighting": random.randint(0, 1),
                #"detail": random.randint(0, 1),
                #"crispness": random.randint(0, 1),
                #"quality": random.randint(0, 1)
            }
        
        # Get object info including input files
        object_info = get_object(object_type)
        
        # ControlNet setup
        if object_info["input_files"]:
            input_img_name = random.choice(object_info["input_files"])
            logger.info(f"run.main: Selected input image {input_img_name} from available files: {object_info['input_files']}")
        else:
            logger.warning(f"run.main: No input files found for object type {object_type}")
            input_img_name = None

        if get_node_ID(workflow, "net1") is not None and input_img_name:
            logger.info(f"run.main: Setting input image to {input_img_name}")
            set_node_value(workflow, "Load Image", "image", input_img_name)

            set_node_value(workflow, "\ud83d\udd79\ufe0f CR Multi-ControlNet Stack", "switch_1", "On")
            set_node_value(workflow, "\ud83d\udd79\ufe0f CR Multi-ControlNet Stack", "switch_2", "On")

        logger.info(f"===== run.main: Running iteration {j} =====")
        style_name = random.choice(art_styles)['name'] if use_art_style else None
        logger.info(f"run.main: Style name: {style_name}")

        for i in range(1, 2):
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
            set_KSampler(workflow, nodeTitle="KSampler", seed=seed, steps=30, cfg=6, sampler_name='dpmpp_2m', scheduler='karras', denoise=1)
            set_KSampler(workflow, nodeTitle="KS_up", seed=seed, steps=10, cfg=4, sampler_name='dpmpp_2m', scheduler='karras', denoise=0.6)
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
                time.sleep(sleep_time_up if run_with_upscale else sleep_time_regular)
    
    # Save the updated workflow
    with open(get_path('workflow', 'randomizer_updated.json'), 'w') as outfile:
        json.dump(workflow, outfile, indent=4)
        outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')