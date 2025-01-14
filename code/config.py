import os

# Get base directories relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up one level from code/

# Define paths relative to BASE_DIR
PATHS = {
    'workflow': os.path.join(BASE_DIR, 'workflow'),
    'res': os.path.join(BASE_DIR, 'res'),
}

def get_path(category, filename):
    """Get full path for a file in a category directory"""
    return os.path.join(PATHS[category], filename) 