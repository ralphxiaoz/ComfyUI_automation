import json
import random
from urllib import request
import csv
import urllib.error
import time
from datetime import datetime


def get_node_ID(workflow, title):
    """
    Retrieves the node ID for a given node title in the workflow.

    Parameters:
    - workflow (dict): The workflow dictionary containing nodes.
    - title (str): The title of the node to find.

    Returns:
    - str: The node ID if found, otherwise None.

    Raises:
    - ValueError: If multiple nodes with the same title are found.
    """
    matching_ids = [node_id for node_id, node in workflow.items() if node.get('_meta', {}).get('title') == title]
    if len(matching_ids) > 1:
        raise ValueError(f"Duplicate titles found for '{title}': {matching_ids}")
    return matching_ids[0] if matching_ids else None


def set_KSampler(workflow, nodeTitle, seed, steps=20, cfg=7, sampler_name='dpmpp_2m', scheduler='karras', denoise=1):
    """
    Configures a KSampler node in the workflow with specified parameters.

    Parameters:
    - workflow (dict): The workflow dictionary containing nodes.
    - nodeTitle (str): The title of the KSampler node to configure.
    - seed (int): The seed value for the sampler.
    - steps (int): The number of steps for the sampler.
    - cfg (int): The configuration value for the sampler.
    - sampler_name (str): The name of the sampler.
    - scheduler (str): The scheduler type.
    - denoise (float): The denoise level.
    """
    node_id = get_node_ID(workflow, nodeTitle)
    if node_id is not None:
        workflow[node_id]['inputs']['seed'] = seed
        workflow[node_id]['inputs']['steps'] = steps
        workflow[node_id]['inputs']['cfg'] = cfg
        workflow[node_id]['inputs']['sampler_name'] = sampler_name
        workflow[node_id]['inputs']['scheduler'] = scheduler
        workflow[node_id]['inputs']['denoise'] = denoise
    else:
        print(f"Node with title '{nodeTitle}' not found.")


def set_lora(workflow, nodeTitle, lora_name, strength_model=1, strength_clip=1):
    """
    Configures a LoRA node in the workflow with specified parameters.

    Parameters:
    - workflow (dict): The workflow dictionary containing nodes.
    - nodeTitle (str): The title of the LoRA node to configure.
    - lora_name (str): The name of the LoRA model.
    - strength_model (float): The model strength.
    - strength_clip (float): The clip strength.
    """
    node_id = get_node_ID(workflow, nodeTitle)
    if node_id is not None:
        workflow[node_id]['inputs']['lora_name'] = lora_name
        workflow[node_id]['inputs']['strength_model'] = strength_model
        workflow[node_id]['inputs']['strength_clip'] = strength_clip
    else:
        print(f"Node with title '{nodeTitle}' not found.")

def update_node_input(workflow, target_title, input_key, new_source_title):
    """
    Updates the input source of a target node in the workflow.

    Parameters:
    - workflow (dict): The workflow dictionary containing nodes.
    - target_title (str): The title of the target node to update.
    - input_key (str): The input key to update.
    - new_source_title (str): The title of the new source node.
    """
    target_node_id = get_node_ID(workflow, target_title)
    new_source_node_id = get_node_ID(workflow, new_source_title)
    
    if target_node_id is not None and new_source_node_id is not None:
        if input_key in workflow[target_node_id]['inputs']:
            workflow[target_node_id]['inputs'][input_key][0] = new_source_node_id
        else:
            print(f"Input key '{input_key}' not found in node '{target_title}'.")
    else:
        if target_node_id is None:
            print(f"Node with title '{target_title}' not found.")
        if new_source_node_id is None:
            print(f"Node with title '{new_source_title}' not found.")


def set_number_of_loras(workflow, num_loras):
    """
    Sets the number of active LoRAs in the workflow and updates connections.

    Parameters:
    - workflow (dict): The workflow dictionary containing nodes.
    - num_loras (int): The number of LoRAs to activate.
    """
    # Identify all LoRA nodes in order
    lora_nodes = sorted([
        node_id for node_id, node in workflow.items() 
        if node.get('class_type') == 'LoraLoader'
        and node.get('_meta', {}).get('title', '').startswith('Lora')  # Only include numbered Loras
    ], key=int)
    
    # Ensure num_loras does not exceed available LoRAs
    num_loras = min(num_loras, len(lora_nodes))
    
    # First, unlink all LoRA nodes
    for lora_id in lora_nodes:
        if 'inputs' in workflow[lora_id]:
            workflow[lora_id]['inputs'] = {
                k: v for k, v in workflow[lora_id]['inputs'].items() 
                if k not in ['model', 'clip']
            }

    # Then set up connections for active LoRAs
    for i in range(num_loras):
        current_lora = lora_nodes[i]
        
        if i == 0:
            # First LoRA connects to checkpoint and CLIP
            workflow[current_lora]['inputs']['model'] = ["1", 0]
            workflow[current_lora]['inputs']['clip'] = ["2", 0]
        else:
            # Other LoRAs connect to previous LoRA
            previous_lora = lora_nodes[i-1]
            workflow[current_lora]['inputs']['model'] = [previous_lora, 0]
            workflow[current_lora]['inputs']['clip'] = [previous_lora, 1]

    # Update all nodes that take input from any LoRA to use the last active LoRA
    if num_loras > 0:
        last_lora = lora_nodes[num_loras - 1]
        for node_id, node in workflow.items():
            if 'inputs' in node:
                for input_key, input_value in node['inputs'].items():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        if input_value[0] in lora_nodes and node_id not in lora_nodes:
                            # Update the connection to point to the last active LoRA
                            workflow[node_id]['inputs'][input_key][0] = last_lora


def output_node_relationship(workflow):
    """
    Outputs the relationships between nodes in the workflow.

    Parameters:
    - workflow (dict): The workflow dictionary containing nodes.
    """
    relationships = {}
    
    for node_id, node in workflow.items():
        node_name = node.get('_meta', {}).get('title', 'Unknown')
        inputs = node.get('inputs', {})
        
        # Create a dictionary for the current node
        relationships[node_id] = {
            'nodeName': node_name,
        }
        
        # Parse only the relevant input structure
        for input_key, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                input_node_id = input_value[0]
                relationships[node_id][f"{input_key}_input"] = input_node_id
                
                # Add output relationship to the input node
                if input_node_id not in relationships:
                    relationships[input_node_id] = {'nodeName': 'Unknown'}
                relationships[input_node_id][f"{input_key}_output"] = node_id
    
    # Format the output
    formatted_output = ', '.join(
        f"{node_id}: {node_info}" for node_id, node_info in relationships.items()
    )
    
    print(f"{{ {formatted_output} }}")


def main():
    """
    Main function to demonstrate the manipulation of a workflow.
    """
    with open('Randomizer.json', 'r') as file:
        workflow = json.load(file)  # JSON data is loaded with double quotes, but Python dictionaries can use single quotes


    # Output the modified workflow
    print("before set_number_of_loras:===")
    with open('randomizer_before_set_loras.json', 'w') as outfile:
        json.dump(workflow, outfile, indent=4)

    set_number_of_loras(workflow, 0)
    
    print("after set_number_of_loras:===")
    with open('randomizer_after_set_loras.json', 'w') as outfile:
        json.dump(workflow, outfile, indent=4)

    print(get_node_ID(workflow, "VAE Decode Up"))

    update_node_input(workflow, "Save Image", "images", "VAE Decode Up")

    print("after update_node_input:===")
    with open('randomizer_after_update_node_input.json', 'w') as outfile:
        json.dump(workflow, outfile, indent=4)

if __name__ == "__main__":
    main()
