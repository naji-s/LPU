import logging
import unittest.mock


import sys
sys.path.append('lpu/external_libs/DEDPUL')
import lpu.external_libs
import lpu.external_libs.DEDPUL
import lpu.external_libs.DEDPUL.NN_functions

import torch
import lpu.utils.utils_general

import lpu.constants
import lpu.datasets.LPUDataset
import lpu.utils.dataset_utils
import lpu.models.dedpul
import lpu.utils.plot_utils
import lpu.utils.utils_general


torch.set_default_dtype(lpu.constants.DTYPE)

LOG = lpu.utils.utils_general.configure_logger(__name__)

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
    'num_epochs': 1000,
    'device': 'cpu',
    'dtype': lpu.constants.DTYPE,
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
        'test': None,
        'val': None,
        'holdout': None,
    },
    'ratios': 
    {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.25,
        'val': 0.2,
        'holdout': .05,
        'train': .5, 
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
    # base_config = lpu.utils.utils_general.load_and_process_config(config['base_config_file_path'])
    # Update the base configuration with the tuning configuration for Ray if it is available
    base_config = lpu.utils.utils_general.deep_update(DEFAULT_CONFIG, config)


    # Initialize training components using the combined configuration
    torch.set_default_dtype(lpu.constants.DTYPE)
    dataloaders_dict = lpu.utils.dataset_utils.create_dataloaders_dict(base_config)
    dedpul_model = lpu.models.dedpul.DEDPUL(base_config)
    
    # Train and report metrics
    scores_dict = dedpul_model.train(
        train_dataloader=dataloaders_dict['train'], 
        val_dataloader=dataloaders_dict['val'], 
        test_dataloader=dataloaders_dict['test'], 
        train_nn_options=base_config['train_nn_options'])
    
    # Flatten scores_dict
    flattened_scores = lpu.utils.utils_general.flatten_dict(scores_dict)
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
