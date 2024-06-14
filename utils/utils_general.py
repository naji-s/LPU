import argparse
import logging
import random

import numpy as np
import torch
import yaml  # Import the yaml module to work with YAML files.

import LPU.constants

def load_and_process_config(yaml_file_path):
    """
    Loads a YAML configuration file, processes it by replacing a specific placeholder with a value,
    and returns the processed YAML content as a Python object.
    
    This function specifically looks for the placeholder 'constants.DTYPE' within the YAML file,
    replaces it with the actual data type defined in 'LPU.constants.DTYPE', and then safely loads
    the YAML content to a Python dictionary or list, depending on the structure of the YAML.
    
    The replacement allows for dynamic configuration values in the YAML file based on the
    'LPU.constants.DTYPE' value at runtime.

    Parameters:
    - yaml_file_path (str): The file path to the YAML configuration file.
    
    Returns:
    - dict or list: The processed YAML content as a Python object. The exact type depends on the
      YAML content structure.
    """
    # Open the YAML file in read mode.
    with open(yaml_file_path, 'r') as file:
        # Read the entire content of the file into a string.
        yaml_content = file.read()
    
    # Replace the 'constants.DTYPE' placeholder with the string representation of 'LPU.constants.DTYPE'.
    # The 'repr()' function is used to get a string representation of 'LPU.constants.DTYPE' that is safe
    # to be evaluated by 'yaml.safe_load' later.
    yaml_content = yaml.safe_load(yaml_content.replace("constants.DTYPE", repr(LPU.constants.DTYPE)))
    
    # Return the processed YAML content. The 'yaml.safe_load' function safely parses the YAML string
    # and returns it as a Python dictionary or list, depending on the YAML structure.
    return yaml_content


def set_seed(seed):
    """
    Set the random seed for reproducible results.
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA devices
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    np.random.seed(seed)  # For NumPy
    random.seed(seed)  # For Python's `random` module
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def configure_logger(module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(LPU.constants.LOG_LEVEL)
    # formatter = logging.Formatter('[[ LOGGING %(asctime)s - %(name)s - %(levelname)s ]]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('[[[ --- LOGGING --- %(asctime)s - %(pathname)s.%(name)s - %(levelname)s ]]]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def deep_update(source, updates):
    """
    Recursively update a dictionary with another dictionary.
    This modifies `source` in place.
    """
    for key, value in updates.items():
        if isinstance(value, dict):
            source[key] = deep_update(source.get(key, {}), value)
        else:
            source[key] = value

    # Add this loop to ensure keys not present in `updates` are preserved
    for key in source:
        if key not in updates:
            if isinstance(source[key], dict):
                source[key] = deep_update(source[key], {})

    return source

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def tune_parse_args():
    parser = argparse.ArgumentParser(description="Tune the dedpul model.")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to run.")
    parser.add_argument("--max_num_epochs", type=int, default=200,
                        help="Maximum number of epochs to run.")
    parser.add_argument("--gpus_per_trial", type=int, default=0,
                        help="Number of GPUs per trial.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to store the results.")
    parser.add_argument("--random_state", type=int, default=None,
                        help="Random state to set.")
    return parser.parse_args()