import json
import random
from urllib import request
import csv
import urllib.error
import time
from datetime import datetime


def generate_pos_prompt(type, options, style=None):
    if type not in options:
        raise ValueError(f"Unknown type: {type}")
    
    prompt_parts = []
    quality_modifier = "masterpiece, best quality, 8k, ultra-detailed, very aesthetic, absurdres, newest, score_9, score_8, score_7"
    
    # Add style-specific keywords if style is specified
    if style and style in options:
        for element, data in options[style]['keywords'].items():
            choice_type = data["choice_type"]
            values = data["values"]
            selected = get_selected_values(choice_type, values)
            prompt_parts.extend(selected)
    
    # Add type-specific keywords
    for element, data in options[type]['keywords'].items():
        choice_type = data["choice_type"]
        values = data["values"]
        selected = get_selected_values(choice_type, values)
        prompt_parts.extend(selected)

    return ", ".join(prompt_parts) + ", " + quality_modifier

def get_selected_values(choice_type, values):
    """Helper function to select values based on choice_type"""
    values = [v for v in values if v]  # Filter out empty values
    
    try:
        num_choices = int(choice_type)
        if num_choices > len(values):
            error_msg = f"Cannot select {num_choices} items from {len(values)} available values"
            print(error_msg)
            raise ValueError(error_msg)
        return random.sample(values, num_choices)
    except ValueError:
        if choice_type == "random":
            count = random.randint(1, len(values))
            return random.sample(values, count)
        elif choice_type == "1":
            return [random.choice(values)]
        elif choice_type == "all":
            return values
        else:
            raise ValueError(f"Unknown choice type: {choice_type}")

def load_workflow(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

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

def load_prompt_options(file_path):
    options = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            type_ = row['Type']
            depth = row['Depth']
            keywords = row['Keywords']
            choice_type = row['Choice_Number']
            
            # Filter out empty values from the remaining columns
            values = [v for k, v in row.items() 
                     if k.startswith('value') and v]
            
            # Initialize type dictionary if it doesn't exist
            if type_ not in options:
                options[type_] = {
                    'depth': depth,  # Store depth separately
                    'keywords': {}   # Store keywords in a nested dict
                }
            
            # Add the keywords data
            options[type_]['keywords'][keywords] = {
                "choice_type": choice_type,
                "values": values
            }
    return options

def get_user_choices():
    # Style choice (Depth 0) - Always ask first
    valid_styles = [row['Type'] for row in csv.DictReader(open('prompt_options.csv')) if row['Depth'] == '0']
    print("\nAvailable styles:", ", ".join(valid_styles))
    style = input("Enter style (or press Enter for none): ").strip().lower()
    if style and style not in valid_styles:
        raise ValueError("Invalid style specified")

    # Load prompt options to filter out depth 0 records
    options = load_prompt_options('prompt_options.csv')
    valid_types = [type_ for type_, details in options.items() if details['depth'] != '0']

    print("\nChoose content type:")
    for idx, type_ in enumerate(valid_types, start=1):
        print(f"{idx}. {type_}")

    base_choice = input(f"Enter choice (1-{len(valid_types)}): ").strip()

    if base_choice.isdigit() and 1 <= int(base_choice) <= len(valid_types):
        return valid_types[int(base_choice) - 1], style
    else:
        raise ValueError("Invalid content type choice")


def main(options, prompt_type=None, style=None):
    try:
        # Use provided prompt_type and style instead of asking again
        if prompt_type is None or style is None:
            prompt_type, style = get_user_choices()
        
        random_seed = random.randint(0, 2**32 - 1)
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Generate prompt using the existing generate_pos_prompt function
        pos_prompt_text = generate_pos_prompt(prompt_type, options, style)
        
        # Base configuration
        ckpt_name = "photonium_v10.safetensors"
        lora_1_name = "dm_canvas_style3_1530.safetensors"
        lora_2_name = "WaterColoursScenev1.0.safetensors"
        
        lora_1_strength_model = 0.75
        lora_1_strength_clip = 1
        lora_2_strength_model = 0
        lora_2_strength_clip = 0
        seed = random_seed

        KS_1_cfg = 7
        KS_1_steps = 20
        KS_2_cfg = 7
        # Resolution
        # 16:9: 3840x2160, 1920x1080, 1280x768, 960x540, 912x512
        # 4:3: 1024x768, 800x600
        width = 912
        height = 512

        width_up = 3840
        height_up = 2160
   
        # Choose workflow based on style first, then type
        if style in ["shuimo", "pixel"]:
            workflow = load_workflow('workflow_genUpLat_API.json')
            workflow["1"]["inputs"]["ckpt_name"] = ckpt_name
            workflow["12"]["inputs"]["lora_name"] = lora_1_name
            workflow["12"]["inputs"]["strength_model"] = lora_1_strength_model
            workflow["12"]["inputs"]["strength_clip"] = lora_1_strength_clip
            workflow["3"]["inputs"]["text"] = pos_prompt_text
            workflow["5"]["inputs"]["seed"] = seed
            workflow["6"]["inputs"]["width"] = width
            workflow["6"]["inputs"]["height"] = height
            workflow["23"]["inputs"]["filename_prefix"] = f"{style}_{prompt_type}_{current_time}"
        else:  # Default to abstract workflow for other cases
            workflow = load_workflow('workflow_gen_API.json')
            workflow["1"]["inputs"]["ckpt_name"] = ckpt_name
            workflow["19"]["inputs"]["lora_name"] = lora_1_name
            workflow["19"]["inputs"]["strength_model"] = lora_1_strength_model
            workflow["19"]["inputs"]["strength_clip"] = lora_1_strength_clip
            workflow["4"]["inputs"]["lora_name"] = lora_2_name
            workflow["4"]["inputs"]["strength_model"] = lora_2_strength_model
            workflow["4"]["inputs"]["strength_clip"] = lora_2_strength_clip
            workflow["2"]["inputs"]["text"] = pos_prompt_text
            workflow["6"]["inputs"]["seed"] = seed
            workflow["6"]["inputs"]["cfg"] = KS_1_cfg
            workflow["6"]["inputs"]["steps"] = KS_1_steps
            workflow["7"]["inputs"]["width"] = width
            workflow["7"]["inputs"]["height"] = height
            """
            # must use upscale variables when using gen&up workflow
            workflow["21"]["inputs"]["width"] = width_up
            workflow["21"]["inputs"]["height"] = height_up
            """
            workflow["11"]["inputs"]["filename_prefix"] = f"{prompt_type}_{current_time}"

        print(pos_prompt_text)
        queue_prompt(workflow)

    except ValueError as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    options = load_prompt_options('prompt_options.csv')

    try:
        # Get user choices once before the loop
        prompt_type, style = get_user_choices()
        num_runs = int(input("Enter the number of times to run the main function: "))
        
        for i in range(num_runs):
            # Pass the pre-selected choices to main
            main(options, prompt_type=prompt_type, style=style)
            print(f"Prompt {i+1}/{num_runs} generated and sent to API")
            if (i + 1) % 10 == 0:
                time.sleep(60)
    except ValueError:
        print("Please enter a valid integer.") 