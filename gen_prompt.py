import csv
import random
import json
from datetime import datetime
from urllib import request
import urllib.error
import time

def gen_trigger_words(ckpt_name, lora_names):
    """
    Generates trigger words for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.

    Returns:
    - str: A comma-separated string of trigger words.
    """
    print(f"\ngen_prompt.gen_trigger_words(): Starting with checkpoint={ckpt_name}, loras={lora_names}")
    
    if isinstance(lora_names, str):
        lora_names = [name.strip() for name in lora_names.split(',')]
        print(f"gen_prompt.gen_trigger_words(): Converted lora_names string to list: {lora_names}")

    def get_keywords(name, type_):
        print(f"gen_prompt.get_keywords(): Searching for {type_}: {name}")
        with open('models.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Name'] == name and row['Type'] == type_:
                    trigger_select = row['Trigger_select']
                    if not trigger_select:
                        print(f"gen_prompt.get_keywords(): No trigger_select found for {name}")
                        return []
                    
                    trigger_keywords = row['Trigger'].split(',')
                    print(f"gen_prompt.get_keywords(): Found triggers for {name}: {trigger_keywords}")
                    
                    if trigger_select.isdigit():
                        num_choices = int(trigger_select)
                        selected = random.sample(trigger_keywords, num_choices)
                        print(f"gen_prompt.get_keywords(): Selected {num_choices} triggers: {selected}")
                        return selected
                    elif trigger_select == "random":
                        count = random.randint(2, len(trigger_keywords))  # More than 1
                        print("gen_prompt.get_keywords: trigger_keywords:", trigger_keywords)
                        return random.sample(trigger_keywords, count)
                    elif trigger_select == "all":
                        print("gen_prompt.get_keywords: trigger_keywords:", trigger_keywords)
                        return trigger_keywords
                    else:
                        raise ValueError("Invalid trigger selection type:", trigger_select)
            print(f"gen_prompt.get_keywords(): No matching model found for {name}")
            return []

    # Concatenate keywords for checkpoint and lora
    checkpoint_keywords = get_keywords(ckpt_name, "Checkpoint")
    lora_keywords = []
    for lora_name in lora_names:
        lora_keywords.extend(get_keywords(lora_name, "Lora"))
    
    result = ', '.join(checkpoint_keywords + lora_keywords)
    print(f"gen_prompt.gen_trigger_words(): Final result: {result}")
    return result

def gen_style_prompt(style_name=None):
    """
    Generates a style prompt by reading from art_styles.csv.

    Parameters:
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - dict: A dictionary with 'positive' and 'negative' prompts for the selected style.
    
    Raises:
    - ValueError: If the specified style_name is not found in the CSV.
    """
    print(f"\ngen_prompt.gen_style_prompt(): Starting with style_name={style_name}")
    
    with open('art_styles.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        
        if style_name:
            for row in rows:
                if row['name'].lower() == style_name.lower():
                    result = {
                        'name': row['name'],
                        'positive': row['positive_prompt'] or "No positive prompt available.",
                        'negative': row['negative_prompt'] or "No negative prompt available."
                    }
                    print(f"gen_prompt.gen_style_prompt(): Found style. Result: {result}")
                    return result
            print(f"gen_prompt.gen_style_prompt(): Style '{style_name}' not found!")
            raise ValueError(f"Style '{style_name}' not found in art_styles.csv")
        else:
            random_row = random.choice(rows)
            return {
                'name': random_row['name'],
                'positive': random_row['positive_prompt'] if random_row['positive_prompt'] else "No positive prompt available.",
                'negative': random_row['negative_prompt'] if random_row['negative_prompt'] else "No negative prompt available."
            }

def gen_positive_prompt(ckpt_name, lora_names, style_name=None):
    """
    Generates a positive prompt for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - str: A comma-separated string of the positive prompt.
    """
    trigger_words = gen_trigger_words(ckpt_name, lora_names)
    print("gen_prompt.py: trigger words of", ckpt_name, lora_names, trigger_words)
    object_string = "1girl, cowboy shot"  # Placeholder for the object
    style_prompt = gen_style_prompt(style_name)
    style_positive = style_prompt['positive']
    quality_modifiers = "masterpiece, best quality, ultra-detailed"  # Placeholder for quality modifiers
    full_prompt = ', '.join([object_string, trigger_words, style_positive, quality_modifiers])
    print("gen_prompt.py: full positive prompt:", full_prompt)
    return full_prompt

def gen_negative_prompt(ckpt_name, lora_names, style_name=None):
    """
    Generates a negative prompt for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - str: A comma-separated string of the negative prompt.
    """
    style_prompt = gen_style_prompt(style_name)
    style_negative = style_prompt['negative']
    quality_modifiers = "embedding:EasyNegative, bad quality, low quality, low resolution"  # Placeholder for quality modifiers

    return ', '.join([style_negative, quality_modifiers])

def main():
    ckpt_name = "duchaitenPonyXLNo_v70.safetensors"
    lora_names = ["Blue_Future.safetensors", "g0th1c2XLP.safetensors", "tifa_v2.3.safetensors"]
    try:
        print("***Positive Prompt***:", gen_positive_prompt(ckpt_name, lora_names, ""))
        print("===Negative Prompt===:", gen_negative_prompt(ckpt_name, lora_names, ""))
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
