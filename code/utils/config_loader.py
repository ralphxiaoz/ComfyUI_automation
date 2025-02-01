import yaml
import os
from pathlib import Path
from config import get_path
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

class ConfigLoader:
    def __init__(self):
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        self.base_config = None

    def load_base_config(self):
        """Load the base configuration file"""
        base_config_path = os.path.join(self.config_dir, 'base_config.yaml')
        try:
            with open(base_config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading base config: {e}")
            raise

    def load_workflow_config(self, workflow_name):
        """Load a specific workflow configuration"""
        if self.base_config is None:
            self.load_base_config()

        workflow_path = os.path.join(self.config_dir, 'workflows', f'{workflow_name}.yaml')
        try:
            with open(workflow_path, 'r') as f:
                workflow_config = yaml.safe_load(f)
                
            # Merge with base config
            config = {
                **self.base_config['common'],
                **workflow_config['settings']
            }
            
            config['workflow_file'] = workflow_config['workflow_file']
            config['name'] = workflow_config['name']
            
            return config
        except Exception as e:
            logger.error(f"Error loading workflow config {workflow_name}: {e}")
            raise 