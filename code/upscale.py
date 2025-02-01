import json
import os
from PIL import Image
from load_models import queue_workflow, set_KSampler
import time
from config import get_path
from utils.logger_config import setup_logger
from node_manipulation import update_node_input, set_resolution, get_node_ID
from datetime import datetime

logger = setup_logger(__name__)

def extract_metadata(image_path):
    """Extract metadata and resolution from an image"""
    try:
        with Image.open(image_path) as img:
            metadata = img.info
            resolution = img.size  # Extract resolution (width, height)
            metadata['resolution'] = resolution
            
            # Try to get workflow from metadata
            workflow_str = metadata.get('prompt')  # ComfyUI stores workflow in 'prompt' field
            if workflow_str:
                try:
                    metadata['workflow'] = json.loads(workflow_str)  # Parse the JSON string
                    return metadata
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse workflow JSON from metadata for {image_path}")
                    return None
            else:
                logger.error(f"No workflow found in metadata for {image_path}")
                return None
                
    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {str(e)}")
        return None

def determine_up_res(base_width, base_height, new_width=None, new_height=None):
    if new_width is not None and new_height is not None:
        return new_width, new_height
    
    aspect_ratio = base_width / base_height
    # Define a tolerance for the aspect ratio
    tolerance = 0.05  # 5% tolerance

    # Check if the aspect ratio is approximately 16:9
    if abs(aspect_ratio - (16 / 9)) < tolerance:
        return 3840, 2160
    else:
        return None

def upscale_images(new_width=None, new_height=None):
    """
    Process images in the to_upscale directory and execute workflows
    
    Parameters:
    - new_width (int): New width for the images
    - new_height (int): New height for the images
    """
    # Use get_path to get the correct to_upscale directory path
    to_upscale = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'to_upscale')
    image_count = 0
    SLEEP_INTERVAL = 5  # Process 5 images before sleeping
    SLEEP_DURATION = 1800  # Sleep for 30 minutes
    
    # Create to_upscale directory if it doesn't exist
    os.makedirs(to_upscale, exist_ok=True)

    for image_file in os.listdir(to_upscale):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(to_upscale, image_file)
            logger.info(f"upscale.process_images: Processing {image_file}")
            
            # Extract metadata including workflow
            metadata = extract_metadata(image_path)
            if not metadata or 'workflow' not in metadata:
                logger.error(f"upscale.process_images: Skipping {image_file} - no valid workflow in metadata")
                continue
            
            workflow = metadata['workflow']
            resolution = metadata['resolution']
            base_width, base_height = resolution

            # Determine new resolution based on provided new_width and new_height
            up_res = determine_up_res(base_width=base_width, base_height=base_height, 
                                    new_width=new_width, new_height=new_height)
            
            if up_res is None:
                logger.error(f"Could not determine upscale resolution for {image_file}")
                continue

            try:
                # Set resolution for upscale nodes - note we're using tuple unpacking here
                width, height = up_res  # Unpack the tuple
                set_resolution(workflow, "Up_res", width=width, height=height)
                
                # Update Save Image node to use upscaled output
                update_node_input(workflow, "Save Image", "images", "VAE Decode_scaled")
                
                # Update output filename to indicate upscaled version
                base_name = os.path.splitext(image_file)[0]
                workflow[get_node_ID(workflow, "Save Image")]["inputs"]["filename_prefix"] = f"{base_name}_upscaled"

                # set the KSampler node values
                set_KSampler(workflow, nodeTitle="KS_up", seed=888, steps=10, cfg=7, sampler_name='dpmpp_2m', scheduler='karras', denoise=0.4)
                
                # Queue the modified workflow
                logger.info(f"upscale.process_images: Queuing workflow for {image_file}")
                queue_workflow(workflow)
                logger.info(f"upscale.process_images: Successfully queued workflow for {image_file}")
                
                # Move processed image to a 'processed' folder
                processed_dir = os.path.join(to_upscale, 'processed')
                os.makedirs(processed_dir, exist_ok=True)
                os.rename(image_path, os.path.join(processed_dir, image_file))
                
                # Increment image counter
                image_count += 1
                
                # Check if we need to sleep
                if image_count % SLEEP_INTERVAL == 0:
                    logger.info(f"Processed {SLEEP_INTERVAL} images. Sleeping for {SLEEP_DURATION/60} minutes...")
                    time.sleep(SLEEP_DURATION)
                    logger.info("Resuming processing...")
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                continue

    # After processing all images, save the last workflow if any were processed
    if image_count > 0:
        # Save the last workflow to a file
        filename = "last_upscale_workflow.json"
        with open(get_path('workflow', filename), 'w') as outfile:
            json.dump(workflow, outfile, indent=4)
            outfile.write(f'\n// Datetime stamp: {datetime.now().isoformat()}\n')
        logger.info(f"Saved last workflow to {filename}")

def main():
    try:
        upscale_images()
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()