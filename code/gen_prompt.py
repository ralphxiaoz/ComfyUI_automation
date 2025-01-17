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

def get_trigger_words(ckpt_name: str, lora_names: Union[str, List[str]], embeddings: Union[str, List[str]]) -> Dict[str, str]:
    """
    Gets positive and negative trigger words for a given checkpoint, LoRAs, and embeddings.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.
    - embeddings (list of str): A list of embedding names.

    Returns:
    - dict: A dictionary with 'positive' and 'negative' trigger words.
    """
    logger.info(f"Getting trigger words - checkpoint: {ckpt_name}, loras: {lora_names}, embeddings: {embeddings}")
    
    
    if isinstance(embeddings, str):
        embeddings = [name.strip() for name in embeddings.split(',')]

    def get_keywords(name, type_, trigger_col, select_col):
        if not name:  # Skip if name is empty
            return []
            
        with open(get_path('res', 'models.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Name'] == name and row['Type'] == type_:
                    trigger_select = row[select_col]
                    if not trigger_select:
                        logger.warning(f"No {select_col} found for {name}")
                        return []
                    
                    trigger_keywords = row[trigger_col].split(',') if row[trigger_col] else []
                    logger.debug(f"Found {trigger_col} for {name}: {trigger_keywords}")
                    
                    if trigger_select.isdigit():
                        num_choices = int(trigger_select)
                        selected = random.sample(trigger_keywords, num_choices) if trigger_keywords else []
                        return selected
                    elif trigger_select == "random":
                        count = random.randint(2, len(trigger_keywords)) if trigger_keywords else 0
                        return random.sample(trigger_keywords, count) if count > 0 else []
                    elif trigger_select == "all":
                        return trigger_keywords
                    else:
                        error_msg = f"Invalid trigger selection type: {trigger_select}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            logger.warning(f"No matching model found for {name}")
            return []

    # Get positive triggers
    pos_checkpoint_keywords = get_keywords(ckpt_name, "Checkpoint", "Pos_trigger", "Pos_trigger_select")
    pos_lora_keywords = []
    for lora_name in lora_names:
        pos_lora_keywords.extend(get_keywords(lora_name, "Lora", "Pos_trigger", "Pos_trigger_select"))
    pos_embedding_keywords = []
    for embedding_name in embeddings:
        pos_embedding_keywords.extend(get_keywords(embedding_name, "Embedding", "Pos_trigger", "Pos_trigger_select"))
    
    # Get negative triggers
    neg_checkpoint_keywords = get_keywords(ckpt_name, "Checkpoint", "Neg_trigger", "Neg_trigger_select")
    neg_lora_keywords = []
    for lora_name in lora_names:
        neg_lora_keywords.extend(get_keywords(lora_name, "Lora", "Neg_trigger", "Neg_trigger_select"))
    neg_embedding_keywords = []
    for embedding_name in embeddings:
        neg_embedding_keywords.extend(get_keywords(embedding_name, "Embedding", "Neg_trigger", "Neg_trigger_select"))
    
    result = {
        'positive': ', '.join(filter(None, pos_checkpoint_keywords + pos_lora_keywords + pos_embedding_keywords)),
        'negative': ', '.join(filter(None, neg_checkpoint_keywords + neg_lora_keywords + neg_embedding_keywords))
    }
    logger.info(f"Final trigger words: {result}")
    return result

def get_style_prompt(style_name=None):
    """
    Gets a style prompt by reading from art_styles.csv.

    Parameters:
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - dict: A dictionary with 'positive' and 'negative' prompts for the selected style.
    
    Raises:
    - ValueError: If the specified style_name is not found in the CSV.
    """
    
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

# def get_object():

def gen_positive_prompt(ckpt_name, lora_names, embeddings, style_name=None):
    """
    Generates a positive prompt for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - str: A comma-separated string of the positive prompt.
    """
    
    object_string = random.choice([
        "1girl, a demon girl with glowing tatoo on skin, smiling at viewer, combat pose", 
        "1girl, a angel girl with huge wings, intricate lingerie, burning heart, kneeing", 
        "1girl, a cat girl, wearing a cat outfit, holding a cat toy, smiling at viewer", 
        "1girl, a demon girl with glowing tatoo on skin, burning hair, full body shot",
        "1mechanical girl,ultra realistic details, portrait, global illumination, shadows, octane render, 8k, ultra sharp,intricate, ornaments detailed, cold colors, metal, egypician detail, highly intricate details, realistic light, trending on cgsociety, glowing eyes, facing camera, neon details, machanical limbs,blood vessels connected to tubes,mechanical vertebra attaching to back,mechanical cervial attaching to neck,sitting,wires and cables connecting to head",
        "complex composition, silhouette of a person sitting on a rock, night scene, vibrant wildflowers, glowing particles, dramatic moonlit sky, realistic textures, intricate details, dynamic lighting, shadows, atmospheric depth, high-quality rendering",
        "1girl, petite, long hair, white hair, (bunny ears:1.075), black bunnysuit, pantyhose, smile, hands on hips, casino"
    ])  # Placeholder for the object

    trigger_words = get_trigger_words(ckpt_name, lora_names, embeddings)
    style_prompt = get_style_prompt(style_name)
    style_positive = style_prompt['positive']
    quality_modifiers = "masterpiece, best quality, ultra-detailed"
    
    full_prompt = ', '.join([object_string, trigger_words['positive'], style_positive, quality_modifiers])
    logger.info(f"Generated positive prompt: {full_prompt}")
    return full_prompt

def gen_negative_prompt(ckpt_name, lora_names, embeddings, style_name=None):
    """
    Generates a negative prompt for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - str: A comma-separated string of the negative prompt.
    """
    
    trigger_words = get_trigger_words(ckpt_name, lora_names, embeddings)
    style_prompt = get_style_prompt(style_name)
    style_negative = style_prompt['negative']
    quality_modifiers = "bad quality, low quality, low resolution"
    
    components = [trigger_words['negative'], style_negative, quality_modifiers]
    full_prompt = ', '.join(component for component in components if component)
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
