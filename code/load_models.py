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
from config import get_path
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def assemble_loras(checkpoint, fixed_loras, lora_categories):
    """
    Assembles a list of LoRAs based on the specified checkpoint, fixed LoRAs, and category quantities.
    This function will only select LoRAs that have the same base as the checkpoint.
    If fixed loras don't have the same base as the checkpoint, they will be ignored.

    Parameters:
    - checkpoint (str): The name of the checkpoint to use.
    - fixed_loras (str or list): A string of fixed LoRAs separated by commas or a list of fixed LoRAs.
    - lora_categories (dict): A dictionary specifying the number of LoRAs to select from each category.

    Returns:
    - list: A list of selected LoRAs.
    """
    # Handle both string and list inputs for fixed_loras
    if isinstance(fixed_loras, str):
        fixed_loras_list = [lora.strip() for lora in fixed_loras.split(',')]
    elif isinstance(fixed_loras, list):
        fixed_loras_list = fixed_loras
    else:
        fixed_loras_list = []

    selected_loras = []

    # Read all rows from CSV file first
    with open(get_path('res', 'models.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        all_models = list(reader)  # Store all rows in memory

    # Find checkpoint and its base
    ckpt_row = next((row for row in all_models if row['Name'] == checkpoint), None)
    if not ckpt_row:
        logger.warning(f"No checkpoint found: {checkpoint}")
        return []

    base = ckpt_row['Base']
    logger.info(f"load_models.assemble_loras: Found checkpoint {checkpoint} with base {base}")

    # Filter LoRAs by base and excluded status
    available_loras = [
        row for row in all_models 
        if row['Type'] == 'Lora' 
        and row['Base'] == base 
        and row['Excluded'] != 'Y'
    ]

    # Add fixed loras first, only add if fixed lora is in available_loras
    for lora_name in fixed_loras_list:
        matching_lora = next((lora for lora in available_loras if lora['Name'] == lora_name), None)
        if matching_lora:
            selected_loras.append(matching_lora['Name'])  # Only append the name
            logger.info(f"load_models-assemble_loras: Added fixed LoRA: {lora_name}")
        else:
            logger.warning(f"load_models.assemble_loras: Fixed LoRA {lora_name} not in available_loras")

    # Add flexible loras based on categories
    for category, quantity in lora_categories.items():
        category_models = [lora for lora in available_loras if lora['Category'] == category]
        
        if not category_models:
            logger.warning(f"No LoRAs found for category: {category}")
            continue

        if quantity == 'all':
            selected_loras.extend([model['Name'] for model in category_models])  # Only extend with names
            logger.info(f"load_models.assemble_loras: Added all {len(category_models)} LoRAs from category {category}")
        elif quantity == 'random':
            if category_models:
                random_count = random.randint(1, len(category_models))
                selected = random.sample(category_models, random_count)
                selected_loras.extend([model['Name'] for model in selected])  # Only extend with names
                logger.info(f"load_models.assemble_loras: Added {len(selected)} random LoRAs from category {category}")
        else:
            try:
                count = int(quantity)
                if category_models:
                    selected = random.sample(category_models, min(count, len(category_models)))
                    selected_loras.extend([model['Name'] for model in selected])  # Only extend with names
                    logger.info(f"load_models.assemble_loras: Added {len(selected)} LoRAs from category {category}")
            except ValueError:
                logger.error(f"load_models.assemble_loras: Invalid quantity value for category {category}: {quantity}")

    # Remove duplicates while preserving order
    seen = set()
    selected_loras = [x for x in selected_loras if not (x in seen or seen.add(x))]
    
    logger.info(f"load_models.assemble_loras: Final selection: {len(selected_loras)} LoRAs")
    for lora in selected_loras:
        # Find the category for logging
        lora_info = next((row for row in available_loras if row['Name'] == lora), None)
        category = lora_info['Category'] if lora_info else 'Unknown'
        logger.info(f"load_models-assemble_loras: Selected LoRA: {lora} (Category: {category})")
    
    return selected_loras

def get_model_params(num_loras, checkpoint=None, loras=None, embeddings=None, skip_external=True):
    """
    Gets model parameters by selecting a checkpoint and a specified number of LoRAs from a CSV file.

    Parameters:
    - num_loras (int): The number of LoRAs to select.
    - checkpoint (str, optional): The name of a specific checkpoint to use. If None, a random checkpoint is selected.
    - loras (list of str, optional): A list of specific LoRA names to use. If None, random LoRAs are selected based on the checkpoint's base.
    - embeddings (list of str, optional): A list of specific embedding names to use. If None, random embeddings are selected based on the checkpoint's base.
    - skip_external (bool): If True, ignore models with a "location" of "external".

    Returns:
    - dict: A dictionary containing the selected checkpoint and LoRAs, along with their associated recommended attributes.
    """
    models = []

    # Read the CSV file
    with open(get_path('res', 'models.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            models.append(row)

    # Filter out Flux.1 D base models and excluded ones
    models = [model for model in models if model['Base'] != 'Flux.1 D' and model['Excluded'] != 'Y']

    # Optionally filter out external models
    if skip_external:
        models = [model for model in models if model.get('Location', '').lower() != 'external']

    # If checkpoint or loras are specified, use them
    selected_checkpoint = None
    selected_loras = []
    selected_embeddings = []

    if checkpoint:
        selected_checkpoint = next((model for model in models if model['Type'] == 'Checkpoint' and model['Name'] == checkpoint), None)
        logger.info(f"load_models.get_model_params: Selected checkpoint: {selected_checkpoint['Name']}")

    if loras:
        selected_loras = [model for model in models if model['Type'] == 'Lora' and model['Name'] in loras]
        logger.info(f"load_models.get_model_params: Selected loras: {selected_loras}")

    if embeddings:
        selected_embeddings = [model for model in models if model['Type'] == 'Embedding' and model['Name'] in embeddings]
        logger.info(f"load_models.get_model_params: Selected embeddings: {selected_embeddings}")

    if not selected_checkpoint:
        # Randomly select a checkpoint if none specified
        checkpoints = [model for model in models if model['Type'] == 'Checkpoint']
        selected_checkpoint = random.choice(checkpoints)
        logger.info(f"load_models.get_model_params: No checkpoint specified, selected random checkpoint: {selected_checkpoint['Name']}")

    if not selected_loras:
        # Filter LoRAs based on the selected checkpoint's base if none specified
        loras = [model for model in models if model['Type'] == 'Lora' and model['Base'] == selected_checkpoint['Base']]
        # Randomly select the specified number of LoRAs
        selected_loras = random.sample(loras, min(num_loras, len(loras)))
        logger.info(f"load_models.get_model_params: No loras specified, selected random loras: {selected_loras}")

    if not selected_embeddings:
        # Get all embeddings that match the checkpoint's base
        embeddings = [model for model in models if model['Type'].startswith('Embedding') and model['Base'] == selected_checkpoint['Base']]
        # Use all available embeddings
        selected_embeddings = random.sample(embeddings, random.randint(0, len(embeddings)-1))
        logger.info(f"load_models.get_model_params: No embeddings specified, selected random embeddings: {selected_embeddings}")

    # Prepare the result dictionary
    result = {
        'checkpoint': selected_checkpoint['Name'],
        'ckpt_clip_skip': selected_checkpoint['Clip_skip'],
        'ckpt_cfg': selected_checkpoint['CFG'],
        'ckpt_steps': selected_checkpoint['Steps'],
        'ckpt_sampler': selected_checkpoint['Sampler'],
        'ckpt_scheduler': selected_checkpoint['Scheduler'],
        'ckpt_hires_fix': selected_checkpoint['HiRes_fix'],
        'num_loras': len(selected_loras),
        'num_embeddings': len(selected_embeddings)
    }

    # Add selected LoRAs to the result
    for i, lora in enumerate(selected_loras, start=1):
        result[f'lora{i}'] = lora['Name']
        result[f'lora{i}_weight_from'] = lora['Weight_from']
        result[f'lora{i}_weight_to'] = lora['Weight_to']
        result[f'lora{i}_clip_skip'] = lora['Clip_skip']
        result[f'lora{i}_cfg'] = lora['CFG']
        result[f'lora{i}_steps'] = lora['Steps']
        result[f'lora{i}_sampler'] = lora['Sampler']
        result[f'lora{i}_scheduler'] = lora['Scheduler']
        result[f'lora{i}_hires_fix'] = lora['HiRes_fix']

    for i, embedding in enumerate(selected_embeddings, start=1):
        result[f'embedding{i}'] = embedding['Name']

    logger.info(f"===== Selected models & parameters: \n {result} =====")
    return result

def load_models_into_workflow(workflow, models):
    """
    Loads a checkpoint and LoRAs into the workflow based on the provided models.

    Parameters:
    - workflow (dict): The workflow dictionary to update.
    - models (dict): The dictionary containing the selected checkpoint and LoRAs.
    """
    # Set the checkpoint
    checkpoint_node_id = get_node_ID(workflow, "Load Checkpoint")
    if checkpoint_node_id:
        workflow[checkpoint_node_id]['inputs']['ckpt_name'] = models['checkpoint']

    # Set the number of LoRAs
    num_loras = len([key for key in models if key.startswith('lora') and not key.endswith(('weight_from', 'weight_to', 'clip_skip', 'cfg', 'steps', 'sampler', 'scheduler', 'hires_fix'))])
    set_number_of_loras(workflow, num_loras)

    # Set each LoRA
    for i in range(1, num_loras + 1):
        lora_name = models[f'lora{i}']
        lora_node_title = f'Lora{i}'
        weight_from = models[f'lora{i}_weight_from']
        weight_to = models[f'lora{i}_weight_to']
        
        # Use 1 by default if weight_from or weight_to are blank
        weight_from = float(weight_from) if weight_from else 1.0
        weight_to = float(weight_to) if weight_to else 1.0
        
        weight = round(random.uniform(weight_from, weight_to), 1)
        set_lora(workflow, lora_node_title, lora_name, strength_model=weight)

def queue_workflow(workflow):

    w = {"prompt": workflow}
    data = json.dumps(w).encode('utf-8')
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            req = request.Request("http://127.0.0.1:8188/prompt", data=data)
            response = request.urlopen(req)
            logger.info(f"load_models.queue_workflow: Response: {response.read().decode()}")
            return True
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP Error (attempt {attempt + 1}/{max_retries}): {e.code} {e.reason}")
            print("Response:", e.read().decode())
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}/{max_retries}):", str(e))
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return False

def main():
    # Example parameters for testing
    checkpoint = "meinamix_v12Final.safetensors"
    fixed_loras = "tifa_v2.3.safetensors, genshinfull1-000006.safetensors"
    lora_categories = {
        "style": 2,
        "detail": "all",
        "quality": "random"
    }

    # Call the assemble_loras function
    selected_loras = assemble_loras(checkpoint, fixed_loras, lora_categories)

    # Log the results
    if selected_loras:
        print("\nSelected LoRAs:")
        for lora in selected_loras:
            print(f"- {lora}")
    else:
        print("No LoRAs selected")

if __name__ == "__main__":
    main()
