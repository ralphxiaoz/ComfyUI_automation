# ComfyUI Automation Framework

This project provides a comprehensive automation framework for ComfyUI, allowing users to easily create, manage, and execute Stable Diffusion workflows programmatically. It focuses on randomizing and managing various aspects of the image generation process, including prompt generation, model loading, and workflow execution.

## Features

- **Dynamic Prompt Generation**: Create rich prompts by combining objects, styles, and model-specific trigger words
- **Model Management**: Automatically load and configure checkpoints, LoRAs, and VAEs
- **Workflow Manipulation**: Programmatically create and modify ComfyUI workflows
- **Randomization**: Generate variations with randomized parameters, LoRAs, and prompts
- **ControlNet Integration**: Support for ControlNet-guided image generation
- **Batch Processing**: Process multiple images with configurable parameters
- **Post-Processing**: Upscale and tweak generated images

## Installation

1. Clone this repository to your local machine
2. Ensure you have ComfyUI installed and properly configured
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── code/                     # Main Python modules
│   ├── config.py             # Configuration handling
│   ├── gen_prompt.py         # Prompt generation utilities
│   ├── load_models.py        # Model loading and management
│   ├── node_manipulation.py  # ComfyUI node manipulation
│   ├── run.py                # Main execution script
│   ├── tweak.py              # Image tweaking utilities
│   ├── upscale.py            # Image upscaling utilities
│   └── utils/                # Utility modules
│       ├── config_loader.py  # YAML configuration loader
│       ├── logger_config.py  # Logging setup
│       └── verify_models.py  # Model verification
├── res/                      # Resource files
│   ├── art_styles.csv        # Art style definitions
│   ├── models.csv            # Model information
│   └── objects.csv           # Object definitions
├── workflow/                 # ComfyUI workflow JSON files
│   ├── Randomizer.json       # Base randomizer workflow
│   ├── Randomizer_controlNet.json  # ControlNet workflow
│   └── ...                   # Other workflow templates
└── tests/                    # Test files (implied by pytest.ini)
```

## Usage

### Basic Usage

To run the basic image generation with randomization:

```bash
python -m code.run
```

### Upscaling Images

Place images in the `to_upscale` directory and run:

```bash
python -m code.upscale
```

### Tweaking Images

Place images in the `to_tweak` directory and run:

```bash
python -m code.tweak
```

### Verify Models

Verify that all models in the CSV files exist and match specifications:

```bash
python -m code.utils.verify_models
```

## Configuration

### Models Configuration

The `res/models.csv` file defines the models used by the system, including:
- Checkpoints (base models)
- LoRAs (model modifications)
- Embeddings (textual embeddings)

Each model has attributes like base type, trigger words, and recommended parameters.

### Art Styles Configuration

The `res/art_styles.csv` file defines different art styles with positive and negative prompts for each style.

### Objects Configuration

The `res/objects.csv` file defines objects with prompts and associated input files for ControlNet.

## Core Components

### Prompt Generation

The `gen_prompt.py` module generates prompts by combining:
- Object descriptions
- Model-specific trigger words
- Art style prompts
- Quality modifiers

### Model Loading

The `load_models.py` module handles:
- Loading checkpoints
- Assembling LoRAs based on categories and compatibility
- Managing model parameters
- Queuing workflows to ComfyUI

### Node Manipulation

The `node_manipulation.py` module provides utilities to:
- Find nodes by title
- Set node values
- Configure samplers and LoRAs
- Set prompts and resolutions
- Update node connections

## Development

To set up the development environment:

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Run tests:
   ```bash
   python run_tests.py
   ```

## Requirements

- Python 3.8+
- ComfyUI (properly configured with models)
- Required Python packages (see `setup.py`)

## License

[Implied open-source license - specific license not provided in files]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
