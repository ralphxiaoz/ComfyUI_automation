import json
import os
from PIL import Image
from load_models import queue_workflow, load_models_into_workflow, get_model_params
import time
from config import get_path
from utils.logger_config import setup_logger
from node_manipulation import update_node_input, set_resolution, get_node_ID, set_lora
from datetime import datetime
from upscale import extract_metadata
import csv

logger = setup_logger(__name__)

def get_loras_from_workflow(workflow):
    """Extract LoRA information from workflow"""
    loras = []
    for node_id, node in workflow.items():
        if node.get('_meta', {}).get('title', '').startswith('Lora'):
            lora_info = {
                'name': node['inputs'].get('lora_name', ''),
                'strength_model': node['inputs'].get('strength_model', 1),
                'strength_clip': node['inputs'].get('strength_clip', 1),
                'node_id': node_id,
                'node_title': node['_meta']['title']
            }
            loras.append(lora_info)
    return loras

def get_lora_params(lora_name):
    """Get LoRA parameters from models.csv"""
    with open(get_path('res', 'models.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Type'] == 'Lora' and row['Name'] == lora_name:
                return {
                    'weight_from': float(row['Weight_from']) if row['Weight_from'] else 1.0,
                    'weight_to': float(row['Weight_to']) if row['Weight_to'] else 1.0
                }
    return None

def tweak_image(image_path, num_tweaks=5):
    """
    Tweak an image by adjusting LoRA weights and generate variations.
    
    Parameters:
    - image_path: Path to the image to tweak
    - num_tweaks: Number of variations to generate (default: 5)
    """
    logger.info(f"tweak.tweak_image: Processing {image_path}")
    
    # Extract metadata including workflow
    metadata = extract_metadata(image_path)
    if not metadata or 'workflow' not in metadata:
        logger.error(f"tweak.tweak_image: No valid workflow found in metadata for {image_path}")
        return
    
    # Get original workflow and extract LoRA information
    workflow = metadata['workflow']
    original_loras = get_loras_from_workflow(workflow)
    
    if not original_loras:
        logger.error(f"tweak.tweak_image: No LoRAs found in workflow for {image_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '98-Tweaked')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Generate variations
    for tweak_num in range(num_tweaks):
        # Create a new workflow for each tweak
        tweaked_workflow = workflow.copy()
        
        # Adjust each LoRA's weights
        for lora in original_loras:
            params = get_lora_params(lora['name'])
            if params:
                # Calculate new weight based on original weight and allowed range
                current_weight = lora['strength_model']
                weight_range = params['weight_to'] - params['weight_from']
                step = weight_range / (num_tweaks + 1)
                new_weight = params['weight_from'] + (step * (tweak_num + 1))
                new_weight = round(new_weight, 2)
                
                # Update LoRA in workflow
                set_lora(tweaked_workflow, lora['node_title'], lora['name'], 
                        strength_model=new_weight, strength_clip=new_weight)
                
                logger.info(f"tweak.tweak_image: Adjusted {lora['name']} weight from {current_weight} to {new_weight}")
        
        # Update output filename
        save_node_id = get_node_ID(tweaked_workflow, "Save Image")
        if save_node_id:
            tweaked_workflow[save_node_id]["inputs"]["filename_prefix"] = f"{base_filename}_tweaked_{tweak_num+1}"
        
        # Queue the workflow
        logger.info(f"tweak.tweak_image: Queuing tweaked workflow {tweak_num+1}")
        queue_workflow(tweaked_workflow)
        
        # Wait between generations
        time.sleep(30)  # Adjust sleep time as needed
        
        # Save the tweaked workflow for reference
        workflow_filename = f"tweaked_workflow_{base_filename}_{tweak_num+1}.json"
        with open(get_path('workflow', workflow_filename), 'w') as outfile:
            json.dump(tweaked_workflow, outfile, indent=4)
            outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')

def process_directory():
    """Process all images in the to_tweak directory"""
    processing_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'to_tweak')
    
    if not os.path.exists(processing_dir):
        logger.error("tweak.process_directory: Processing directory does not exist")
        return
    
    for image_file in os.listdir(processing_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(processing_dir, image_file)
            tweak_image(image_path)
            
            # Move processed image to a 'processed' subdirectory
            processed_dir = os.path.join(processing_dir, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            os.rename(image_path, os.path.join(processed_dir, image_file))

def main():
    try:
        process_directory()
    except Exception as e:
        logger.error(f"tweak.main: Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()

