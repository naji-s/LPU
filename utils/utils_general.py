import lpu.constants
import numpy as np
import random
import torch
import yaml  # Import the yaml module to work with YAML files.

def load_and_process_config(yaml_file_path):
    """
    Loads a YAML configuration file, processes it by replacing a specific placeholder with a value,
    and returns the processed YAML content as a Python object.
    
    This function specifically looks for the placeholder 'constants.DTYPE' within the YAML file,
    replaces it with the actual data type defined in 'lpu.constants.DTYPE', and then safely loads
    the YAML content to a Python dictionary or list, depending on the structure of the YAML.
    
    The replacement allows for dynamic configuration values in the YAML file based on the
    'lpu.constants.DTYPE' value at runtime.

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
    
    # Replace the 'constants.DTYPE' placeholder with the string representation of 'lpu.constants.DTYPE'.
    # The 'repr()' function is used to get a string representation of 'lpu.constants.DTYPE' that is safe
    # to be evaluated by 'yaml.safe_load' later.
    yaml_content = yaml.safe_load(yaml_content.replace("constants.DTYPE", repr(lpu.constants.DTYPE)))
    
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