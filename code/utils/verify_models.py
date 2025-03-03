import os
import csv
from pathlib import Path
import sys

"""
This script verifies that all models in the CSV exist in the directories and vice versa.
"""

# Add the parent directory to the Python path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_path
from utils.logger_config import setup_logger


logger = setup_logger(__name__)

def get_model_files(base_path, subfolder):
    """Get all model files from a specified directory."""
    model_dir = os.path.join(base_path, 'models', subfolder)
    if not os.path.exists(model_dir):
        logger.error(f"Directory not found: {model_dir}")
        return set()
    
    extensions = {'.safetensors', '.pt', '.ckpt', '.pth'}
    # Convert all filenames to lowercase for case-insensitive comparison
    return {f.name.lower() for f in Path(model_dir).iterdir() 
            if f.is_file() and f.suffix.lower() in extensions}

def get_csv_models():
    """Get model names from CSV file grouped by type."""
    csv_models = {
        'checkpoints': set(),
        'loras': set(),
        'embeddings': set()
    }
    
    try:
        with open(get_path('res', 'models.csv'), 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model_type = row['Type'].lower()
                # Convert model names to lowercase for case-insensitive comparison
                if model_type == 'checkpoint':
                    csv_models['checkpoints'].add(row['Name'].lower())
                elif model_type == 'lora':
                    csv_models['loras'].add(row['Name'].lower())
                elif model_type == 'embedding':
                    csv_models['embeddings'].add(row['Name'].lower())
    except FileNotFoundError:
        logger.error("models.csv not found")
        return csv_models
    
    return csv_models

def verify_models(comfyui_path):
    """Verify that all models in the CSV exist in the directories and vice versa."""
    logger.info("Starting model verification...")
    
    # Get files from directories
    file_models = {
        'checkpoints': get_model_files(comfyui_path, 'checkpoints'),
        'loras': get_model_files(comfyui_path, 'loras'),
        'embeddings': get_model_files(comfyui_path, 'embeddings')
    }
    
    # Get models from CSV
    csv_models = get_csv_models()
    
    # Check each category
    for category in ['checkpoints', 'loras', 'embeddings']:
        logger.info(f"\nChecking {category}...")
        
        # Files that exist but aren't in CSV
        missing_in_csv = file_models[category] - csv_models[category]
        if missing_in_csv:
            logger.warning(f"\nFiles existing but not in CSV ({len(missing_in_csv)}):")
            for model in sorted(missing_in_csv):
                logger.warning(f"- {model}")
        
        # CSV entries that don't exist as files
        missing_files = csv_models[category] - file_models[category]
        if missing_files:
            logger.warning(f"\nCSV entries without files ({len(missing_files)}):")
            for model in sorted(missing_files):
                logger.warning(f"- {model}")
        
        # Matching entries
        matching = file_models[category] & csv_models[category]
        logger.info(f"\nMatching entries: {len(matching)}/{len(csv_models[category])}")

def main():
    comfyui_path = r"C:\Users\leife\Documents\Projects\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI"
    verify_models(comfyui_path)

if __name__ == "__main__":
    main() 