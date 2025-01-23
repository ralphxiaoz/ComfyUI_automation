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
    logger.info(f"gen_prompt.get_trigger_words: Final trigger words: \n {result}")
    return result

def get_style_prompt(style_name=None):
    """
    Gets a style prompt by reading from art_styles.csv.

    Parameters:
    - style_name (str, optional): The name of the style to retrieve.

    Returns:
    - dict: A dictionary with 'positive' and 'negative' prompts for the selected style.
    
    Raises:
    - ValueError: If the specified style_name is not found in the CSV.
    """
    
    with open(get_path('res', 'art_styles.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        
        # Log the number of rows and included styles for debugging
        included_rows = [row for row in rows if row.get('included', '').lower() == 'y']
        logger.info(f"gen_prompt.select_random_style: Total styles: {len(rows)}, Included styles: {len(included_rows)}")
        
        if included_rows == []:
            logger.info("No styles available in art_styles.csv that are marked as included.")
            return {"positive": "", "negative": ""}

        if style_name:
            for row in rows:
                if row['name'].lower() == style_name.lower() and row.get('included', '').lower() == 'y':
                    result = {
                        'name': row['name'],
                        'positive': row['positive_prompt'] or "No positive prompt available.",
                        'negative': row['negative_prompt'] or "No negative prompt available."
                    }
                    logger.info(f"Found style. Result: {result}")
                    return result
            error_msg = f"Style '{style_name}' not found in art_styles.csv or not included."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            """
            # Filter rows to only include those marked with 'y' in the "included" column
            included_rows = [row for row in rows if row.get('included', '').lower() == 'y']
            if not included_rows:
                logger.error("No styles available in art_styles.csv that are marked as included.")
                # Log all unique values in the 'included' column for debugging
                included_values = set(row.get('included', '') for row in rows)
                logger.error(f"Values found in 'included' column: {included_values}")
                raise ValueError("No styles available.")
            random_row = random.choice(included_rows)
            result = {
                'name': random_row['name'],
                'positive': random_row['positive_prompt'] if random_row['positive_prompt'] else "No positive prompt available.",
                'negative': random_row['negative_prompt'] if random_row['negative_prompt'] else "No negative prompt available."
            }
            """
            logger.info("No style selected")
            return {"name":"", "positive": "", "negative": ""}

def get_object(identifier):
    """
    Gets an object prompt by reading from objects.csv.
    Can be called with either a serial number or type.

    Parameters:
    - identifier (str/int): Either a serial number or type to search for

    Returns:
    - dict: A dictionary containing 'name', 'positive', and 'negative' prompts for the selected object
    """
    logger.info(f"gen_prompt.get_object: Getting object with identifier: {identifier}")
    
    try:
        with open(get_path('res', 'objects.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            # Check if identifier is a serial number
            if str(identifier).isdigit():
                matching_rows = [row for row in rows if row['serial_no'] == str(identifier)]
                if not matching_rows:
                    logger.warning(f"gen_prompt.get_object: No object found for serial_no: {identifier}")
                    return {"name": "", "positive": "", "negative": ""}
                selected_row = matching_rows[0]  # Take the exact match
            else:
                # Treat identifier as type
                matching_rows = [row for row in rows if row['type'].lower() == str(identifier).lower()]
                if not matching_rows:
                    logger.warning(f"gen_prompt.get_object: No objects found for type: {identifier}")
                    return {"name": "", "positive": "", "negative": ""}
                selected_row = random.choice(matching_rows)  # Random selection from type
            
            result = {
                "name": f"object_{selected_row['serial_no']}",
                "positive": selected_row['positive_prompt'],
                "negative": selected_row['negative_prompt']
            }
            
            logger.info(f"gen_prompt.get_object: Selected object: {result['name']}")
            return result
            
    except FileNotFoundError:
        logger.error("gen_prompt.get_object: objects.csv not found in res directory")
        return {"name": "", "positive": "", "negative": ""}
    except KeyError as e:
        logger.error(f"gen_prompt.get_object: CSV file missing required column: {e}")
        return {"name": "", "positive": "", "negative": ""}

def gen_positive_prompt(ckpt_name, lora_names, object_type, embeddings, style_name=None):
    """
    Generates a positive prompt for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.
    - object_type (str): The type of object to retrieve.
    - embeddings (list of str): A list of embedding names.
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - str: A comma-separated string of the positive prompt.
    """
    
    object_string = get_object(object_type)['positive']

    trigger_words = get_trigger_words(ckpt_name, lora_names, embeddings)
    style_prompt = get_style_prompt(style_name)
    style_positive = style_prompt['positive']
    quality_modifiers = "masterpiece, best quality, ultra-detailed"
    
    full_prompt = ', '.join([object_string, trigger_words['positive'], style_positive, quality_modifiers])
    logger.info(f"gen_prompt.gen_positive_prompt: Full positive prompt: \n {full_prompt}")
    return full_prompt

def gen_negative_prompt(ckpt_name, lora_names, object_type, embeddings, style_name=None):
    """
    Generates a negative prompt for a given checkpoint and a list of LoRAs.

    Parameters:
    - ckpt_name (str): The name of the checkpoint.
    - lora_names (list of str): A list of LoRA names.
    - style_name (str, optional): The name of the style to retrieve. If None, returns a random style.

    Returns:
    - str: A comma-separated string of the negative prompt.
    """
    object_string = get_object(object_type)['negative']

    trigger_words = get_trigger_words(ckpt_name, lora_names, embeddings)
    style_prompt = get_style_prompt(style_name)
    style_negative = style_prompt['negative']
    quality_modifiers = "watermark, bad quality, low quality, low resolution"
    
    components = [trigger_words['negative'], style_negative, quality_modifiers, object_string]
    full_prompt = ', '.join(component for component in components if component)
    logger.info(f"gen_prompt.gen_negative_prompt: Full negative prompt: \n {full_prompt}")
    return full_prompt

def main():
    ckpt_name = "duchaitenPonyXLNo_v70.safetensors"
    lora_names = []
    try:
        logger.info("Starting prompt generation test")
        pos_prompt = gen_positive_prompt(ckpt_name, lora_names, "target", "")
        neg_prompt = gen_negative_prompt(ckpt_name, lora_names, "")
        logger.info(f"Positive Prompt: {pos_prompt}")
        logger.info(f"Negative Prompt: {neg_prompt}")
    except ValueError as e:
        logger.error(f"Error during prompt generation: {str(e)}")

if __name__ == "__main__":
    main()
