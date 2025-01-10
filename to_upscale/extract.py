from PIL import Image

def extract_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = img.info
        resolution = img.size  # Extract resolution (width, height)
        metadata['resolution'] = resolution  # Add resolution to metadata
    return metadata

# Usage
image_path = "to_upscale/character_sfw_20250102114629_00001_.png"
metadata = extract_metadata(image_path)
print(metadata)
