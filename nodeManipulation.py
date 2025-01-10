import json
import random
from urllib import request
import csv
import urllib.error
import time
from datetime import datetime


def get_node_ID(workflow, title):
    for node_id, node in workflow.items():
        if node.get('_meta', {}).get('title') == title:
            return node_id
    return None


def set_KSampler(workflow, nodeTitle, seed, steps=20, cfg=7, sampler_name='dpmpp_2m', scheduler='karras', denoise=1):
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
    node_id = get_node_ID(workflow, nodeTitle)
    if node_id is not None:
        workflow[node_id]['inputs']['lora_name'] = lora_name
        workflow[node_id]['inputs']['strength_model'] = strength_model
        workflow[node_id]['inputs']['strength_clip'] = strength_clip
    else:
        print(f"Node with title '{nodeTitle}' not found.")


def set_number_of_loras(workflow, num_loras):
    # Identify all LoRA nodes in order
    lora_nodes = sorted([
        node_id for node_id, node in workflow.items() 
        if node.get('class_type') == 'LoraLoader'
        and node.get('_meta', {}).get('title', '').startswith('Lora')  # Only include numbered Loras
    ])
    
    # Ensure num_loras does not exceed available LoRAs
    num_loras = min(num_loras, len(lora_nodes))
    
    # Build a complete map of node relationships
    relationships = {}
    for node_id, node in workflow.items():
        inputs = node.get('inputs', {})
        for input_key, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) == 2:
                input_node_id = input_value[0]
                if input_node_id in lora_nodes:
                    if node_id not in relationships:
                        relationships[node_id] = {}
                    relationships[node_id][input_key] = input_node_id

    # Keep track of nodes to remove
    nodes_to_remove = lora_nodes[num_loras:]

    # Update connections for active LoRAs
    for i, lora_node_id in enumerate(lora_nodes[:num_loras]):
        if i == 0:
            # First LoRA's connections should not be modified
            continue
        elif i < num_loras - 1:
            # Connect this LoRA to the next LoRA
            next_lora = lora_nodes[i + 1]
            workflow[next_lora]['inputs']['model'][0] = lora_node_id
            workflow[next_lora]['inputs']['clip'][0] = lora_node_id

    # Connect all dependent nodes to the last active LoRA
    if num_loras > 0:
        last_lora = lora_nodes[num_loras - 1]
        for dep_node_id, inputs in relationships.items():
            for input_key, input_node in inputs.items():
                if input_node in lora_nodes and dep_node_id not in lora_nodes[:num_loras]:
                    workflow[dep_node_id]['inputs'][input_key][0] = last_lora

    # Remove unused LoRA nodes
    for node_id in nodes_to_remove:
        if node_id in workflow:
            del workflow[node_id]


def output_node_relationship(workflow):
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
    with open('Randomizer.json', 'r') as file:
        workflow = json.load(file)  # JSON data is loaded with double quotes, but Python dictionaries can use single quotes


    # Output the modified workflow
    print("before set_number_of_loras:===")
    with open('randomizer_before_set_loras.json', 'w') as outfile:
        json.dump(workflow, outfile, indent=4)

    set_number_of_loras(workflow, 2)
    
    print("after set_number_of_loras:===")
    with open('randomizer_after_set_loras.json', 'w') as outfile:
        json.dump(workflow, outfile, indent=4)

if __name__ == "__main__":
    main()
