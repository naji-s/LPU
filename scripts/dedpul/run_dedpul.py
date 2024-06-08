import logging
import unittest.mock


import sys
sys.path.append('LPU/external_libs/DEDPUL')
import LPU.external_libs
import LPU.external_libs.DEDPUL
import LPU.external_libs.DEDPUL.NN_functions

import torch
import LPU.utils.utils_general

import LPU.constants
import LPU.datasets.LPUDataset
import LPU.utils.dataset_utils
import LPU.models.dedpul
import LPU.utils.plot_utils
import LPU.utils.utils_general


torch.set_default_dtype(LPU.constants.DTYPE)

LOG = LPU.utils.utils_general.configure_logger(__name__)

# Optional dynamic import for Ray
try:
    import ray.util.client
    import ray.train
    RAY_AVAILABLE = True
except ImportError:
    LOG.warning("Ray is not available. Please install Ray to enable distributed training.")
    RAY_AVAILABLE = False


USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    'learning_rate': .01,
    'num_epochs': 10,
    'device': 'cpu',
    'dtype': LPU.constants.DTYPE,
    'train_val_ratio': .1,
    'evaluation_interval': 1,
    'epoch_blocks': 1,
    'nrep': 10,
    'dedpul_type': 'dedpul',
    'dataset_kind': 'LPU',
    'dataset_name': 'animal_no_animal',
    'cv': 5,
    'estimate_diff_options':
    {
        'MT': True, 
        'MT_coef': 0.25,
        'decay_MT_coef': False,
        'tune': False,
        'bw_mix': 0.05, 
        'bw_pos': 0.1,
        'threshold': 'mid',
        'n_gauss_mix': 20,
        'n_gauss_pos': 10,
        'bins_mix': 20,
        'bins_pos': 20,
        'k_neighbours': None,
    },
    'batch_size': 
    {
        'train': 64,
        'test': 64,
        'val': 64,
        'holdout': 64,
    },
    'ratios': 
    {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.25,
        'val': 0.1,
        'holdout': .0,
        'train': .65, 
    },
    'train_nn_options': 
    { 
        'loss_function': 'log',
        'n_early_stop': 20,
        'disp': False,
    }
}


def train_model(config=None):
    if config is None:
        config = {}
    # Load the base configuration
    # base_config = LPU.utils.utils_general.load_and_process_config(config['base_config_file_path'])
    # Update the base configuration with the tuning configuration for Ray if it is available

    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(DEFAULT_CONFIG, config)
    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
    else:
        random_state = LPU.constants.RANDOM_STATE
        
    LPU.utils.utils_general.set_seed(random_state)


    # Initialize training components using the combined configuration
    torch.set_default_dtype(LPU.constants.DTYPE)
    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    dedpul_model = LPU.models.dedpul.DEDPUL(config)
    
    # Train and report metrics
    scores_dict = dedpul_model.train(
        train_dataloader=dataloaders_dict['train'], 
        val_dataloader=dataloaders_dict['val'], 
        test_dataloader=dataloaders_dict['test'], 
        train_nn_options=config['train_nn_options'])
    
    # Flatten scores_dict
    flattened_scores = LPU.utils.utils_general.flatten_dict(scores_dict)
    filtered_scores_dict = {}
    for key, value in flattened_scores.items():
        if 'train' in key or 'val' in key or 'test' in key:
            if 'epochs' not in key:
                filtered_scores_dict[key] = value
    print("Reporting Metrics: ", filtered_scores_dict)  # Debug print to check keys

    # Report metrics if executed under Ray Tune
    if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
        ray.train.report(filtered_scores_dict)
    else:
        return scores_dict

if __name__ == "__main__":
    # main()
    results = train_model()
