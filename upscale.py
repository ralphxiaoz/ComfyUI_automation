import json
import os
from PIL import Image, ExifTags
from auto_gen import queue_prompt, load_workflow
import time

def extract_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = img.info
        resolution = img.size  # Extract resolution (width, height)
        metadata['resolution'] = resolution  # Add resolution to metadata
    return metadata

def process_images(upscale_by):
    """Process images in the to_upscale directory and execute workflows"""
    to_upscale = './to_upscale'
    image_count = 0
    SLEEP_INTERVAL = 5  # Process 5 images before sleeping
    SLEEP_DURATION = 1800  # Sleep for 30 minutes (30 * 60 seconds)
    
    for image_file in os.listdir(to_upscale):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(to_upscale, image_file)
            
            # Extract metadata
            metadata = extract_metadata(image_path)
            
            # Load the workflow template
            workflow = metadata.get('prompt')
            workflow = json.loads(workflow)
            print("===workflow===", workflow)
            
            # Update the image node number to 15
            if "11" in workflow:
                if "inputs" in workflow["11"] and "images" in workflow["11"]["inputs"]:
                    workflow["11"]["inputs"]["images"][0] = "15"
                    print("Updated image node to 15")
            
            resolution = metadata.get('resolution')
            # Use upscale_by to adjust resolution or other parameters if needed
            if "21" in workflow:
                workflow["21"]["inputs"]["width"] = 3840 #int(resolution[0] * upscale_by)
                workflow["21"]["inputs"]["height"] = 2160 #int(resolution[1] * upscale_by)
                workflow["11"]["inputs"]["filename_prefix"] = f"{os.path.splitext(image_file)[0]}_upscaled"
            else:
                print("No upscale node found in the extracted workflow, aborting")
                continue

            # Queue the modified workflow
            print(f"Queuing workflow for {image_file}...")
            queue_prompt(workflow)
            print(f"Workflow for {image_file} queued successfully")
            
            # Increment image counter
            image_count += 1
            
            # Check if we need to sleep
            if image_count % SLEEP_INTERVAL == 0:
                print(f"\nProcessed {SLEEP_INTERVAL} images. Sleeping for 30 minutes...")
                time.sleep(SLEEP_DURATION)
                print("Resuming processing...")

if __name__ == "__main__":
    try:
        upscale_by = float(input("Enter the upscale factor (e.g., 1.5 for 150%): "))
        process_images(upscale_by)
    except ValueError:
        print("Please enter a valid number for the upscale factor.")