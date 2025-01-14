import csv
import random
import json
from datetime import datetime
from urllib import request
import urllib.error
import time
from config import get_path
from typing import List, Dict, Optional, Union
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def get_trigger_words(ckpt_name: str, lora_names: Union[str, List[str]]) -> str:
    """
    Gets trigger words for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.

    Returns:
    - str: A comma-separated string of trigger words.
    """
    logger.info(f"Getting trigger words - checkpoint: {ckpt_name}, loras: {lora_names}")
    
    if isinstance(lora_names, str):
        lora_names = [name.strip() for name in lora_names.split(',')]
        logger.debug(f"Converted lora_names string to list: {lora_names}")

    def get_keywords(name, type_):
        logger.info(f"Searching for {type_}: {name}")
        with open(get_path('res', 'models.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Name'] == name and row['Type'] == type_:
                    trigger_select = row['Trigger_select']
                    if not trigger_select:
                        logger.warning(f"No trigger_select found for {name}")
                        return []
                    
                    trigger_keywords = row['Trigger'].split(',')
                    logger.debug(f"Found triggers for {name}: {trigger_keywords}")
                    
                    if trigger_select.isdigit():
                        num_choices = int(trigger_select)
                        selected = random.sample(trigger_keywords, num_choices)
                        logger.debug(f"Selected {num_choices} triggers: {selected}")
                        return selected
                    elif trigger_select == "random":
                        count = random.randint(2, len(trigger_keywords))
                        logger.debug(f"Random selection - count: {count}, keywords: {trigger_keywords}")
                        return random.sample(trigger_keywords, count)
                    elif trigger_select == "all":
                        logger.debug(f"Using all triggers: {trigger_keywords}")
                        return trigger_keywords
                    else:
                        error_msg = f"Invalid trigger selection type: {trigger_select}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            logger.warning(f"No matching model found for {name}")
            return []

    checkpoint_keywords = get_keywords(ckpt_name, "Checkpoint")
    lora_keywords = []
    for lora_name in lora_names:
        lora_keywords.extend(get_keywords(lora_name, "Lora"))
    
    result = ', '.join(checkpoint_keywords + lora_keywords)
    logger.info(f"Final trigger words: {result}")
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
    logger.info(f"Generating style prompt - style_name: {style_name}")
    
    with open(get_path('res', 'art_styles.csv'), 'r') as csvfile:
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
                    logger.info(f"Found style. Result: {result}")
                    return result
            error_msg = f"Style '{style_name}' not found in art_styles.csv"
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            random_row = random.choice(rows)
            result = {
                'name': random_row['name'],
                'positive': random_row['positive_prompt'] if random_row['positive_prompt'] else "No positive prompt available.",
                'negative': random_row['negative_prompt'] if random_row['negative_prompt'] else "No negative prompt available."
            }
            logger.info(f"Selected random style: {result['name']}")
            return result

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
    logger.info(f"Generating positive prompt - style: {style_name}")
    trigger_words = get_trigger_words(ckpt_name, lora_names)
    logger.debug(f"Got trigger words: {trigger_words}")
    
    object_string = random.choice([
        "1girl, a demon girl with glowing tatoo on skin, smiling at viewer, combat pose", 
        "1girl, a angel girl with huge wings, intricate lingerie, burning heart, kneeing", 
        "1girl, a cat girl, wearing a cat outfit, holding a cat toy, smiling at viewer", 
        "1girl, a demon girl with glowing tatoo on skin, burning hair, full body shot",
        "1girl, a witch girl, long black hair,intricate outfit, holding a book, cowboy shot" 
    ])  # Placeholder for the object
    style_prompt = gen_style_prompt(style_name)
    style_positive = style_prompt['positive']
    quality_modifiers = "masterpiece, best quality, ultra-detailed"
    
    full_prompt = ', '.join([object_string, trigger_words, style_positive, quality_modifiers])
    logger.info(f"Generated positive prompt: {full_prompt}")
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
    logger.info(f"Generating negative prompt - style: {style_name}")
    
    style_prompt = gen_style_prompt(style_name)
    style_negative = style_prompt['negative']
    quality_modifiers = "embedding:EasyNegative, bad quality, low quality, low resolution"
    
    full_prompt = ', '.join([style_negative, quality_modifiers])
    logger.info(f"Generated negative prompt: {full_prompt}")
    return full_prompt

def main():
    ckpt_name = "duchaitenPonyXLNo_v70.safetensors"
    lora_names = ["Blue_Future.safetensors", "g0th1c2XLP.safetensors", "tifa_v2.3.safetensors"]
    try:
        logger.info("Starting prompt generation test")
        pos_prompt = gen_positive_prompt(ckpt_name, lora_names, "")
        neg_prompt = gen_negative_prompt(ckpt_name, lora_names, "")
        logger.info(f"Positive Prompt: {pos_prompt}")
        logger.info(f"Negative Prompt: {neg_prompt}")
    except ValueError as e:
        logger.error(f"Error during prompt generation: {str(e)}")

if __name__ == "__main__":
    main()
