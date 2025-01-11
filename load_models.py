import csv
import random
import json
from datetime import datetime
from urllib import request
import urllib.error
from node_manipulation import set_number_of_loras, set_lora, get_node_ID

def pick_models(num_loras, checkpoint=None, loras=None):
    """
    Selects a checkpoint and a specified number of LoRAs from a CSV file.

    Parameters:
    - num_loras (int): The number of LoRAs to select.
    - checkpoint (str, optional): The name of a specific checkpoint to use. If None, a random checkpoint is selected.
    - loras (list of str, optional): A list of specific LoRA names to use. If None, random LoRAs are selected based on the checkpoint's base.

    Returns:
    - dict: A dictionary containing the selected checkpoint and LoRAs, along with their associated attributes.
    """
    models = []

    # Read the CSV file
    with open('models.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            models.append(row)

    # Filter out Flux.1 D base models
    models = [model for model in models if model['Base'] != 'Flux.1 D']

    # If checkpoint or loras are specified, use them
    selected_checkpoint = None
    selected_loras = []

    if checkpoint:
        selected_checkpoint = next((model for model in models if model['Type'] == 'Checkpoint' and model['Name'] == checkpoint), None)

    if loras:
        selected_loras = [model for model in models if model['Type'] == 'Lora' and model['Name'] in loras]

    if not selected_checkpoint:
        # Randomly select a checkpoint if none specified
        checkpoints = [model for model in models if model['Type'] == 'Checkpoint']
        selected_checkpoint = random.choice(checkpoints)

    if not selected_loras:
        # Filter LoRAs based on the selected checkpoint's base if none specified
        loras = [model for model in models if model['Type'] == 'Lora' and model['Base'] == selected_checkpoint['Base']]
        # Randomly select the specified number of LoRAs
        selected_loras = random.sample(loras, min(num_loras, len(loras)))

    # Prepare the result dictionary
    result = {
        'checkpoint': selected_checkpoint['Name'],
        'ckpt_clip_skip': selected_checkpoint['Clip_skip'],
        'ckpt_cfg': selected_checkpoint['CFG'],
        'ckpt_steps': selected_checkpoint['Steps'],
        'ckpt_sampler': selected_checkpoint['Sampler'],
        'ckpt_scheduler': selected_checkpoint['Scheduler'],
        'ckpt_hires_fix': selected_checkpoint['HiRes_fix'],
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

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    
    
    req = request.Request("http://127.0.0.1:8188/prompt", data=data)
    try:
        response = request.urlopen(req)
        print("Response:", response.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.reason)
        print("Response:", e.read().decode())

# Example usage
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

    for ckpt in checkpoints:
        n = 2  # Replace with the desired index (1-based) of the checkpoint
        ckpt = checkpoints[n - 1] if 0 < n <= len(checkpoints) else random.choice(checkpoints)
        for i in range(1, 10):
            num_loras = random.randint(1, 4)
            models = pick_models(num_loras, checkpoint=ckpt, loras=None)
            load_models_into_workflow(workflow, models)
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            checkpoint_used = models['checkpoint']
            loras_used = ', '.join([models[f'lora{i}'] for i in range(1, num_loras + 1)])
            workflow["12"]["inputs"]["filename_prefix"] = f"randomizer-{current_time}-{checkpoint_used.replace('.safetensors', '')}-{loras_used.replace('.safetensors', '')}"
            queue_prompt(workflow)

    # Save the updated workflow to a new JSON file
    with open('randomizer_updated.json', 'w') as outfile:
        json.dump(workflow, outfile, indent=4)
        outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')  # Commented out datetime stamp
