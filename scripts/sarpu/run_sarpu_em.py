import logging
import unittest.mock
import numpy as np
import torch
import LPU.constants
import LPU.datasets.LPUDataset
import sys
sys.path.append('LPU/external_libs/SAR_PU')
sys.path.append('LPU/external_libs/SAR_PU/sarpu')
sys.path.append('LPU/external_libs/SAR_PU/sarpu/sarpu')
import LPU.models.SARPU.sarpu_em
import LPU.utils.utils_general
import LPU.utils.dataset_utils
import LPU.utils.plot_utils
import LPU.external_libs.SAR_PU.sarpu.sarpu.PUmodels
import LPU.external_libs.SAR_PU.sarpu.sarpu.pu_learning
import LPU.external_libs.SAR_PU.sarpu
import LPU.external_libs.SAR_PU.sarpu.sarpu

torch.set_default_dtype(LPU.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    "inducing_points_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 5,
    "device": "cpu",
    "epoch_block": 1,
    "dataset_name": "animal_no_animal",  # could also be fashionMNIST
    "dataset_kind": "LPU",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)

    # setting up parameters for the classification model of sar_pu
    "SAR_PU_classification_model": 'logistic',
    "svm_params": 
        {
        'tol': 1e-4,
        'C': 1.0,
        'kernel': 'rbf', 
        'degree': 3, 
        'gamma': 'scale',
        'class_weight': None,
        'random_state': None,  
         'max_iter':-1, 
         'cache_size':200,
         'decision_function_shape': 'ovr', 
         'verbose': 0    
        },
    'logistic_params':
        {'penalty': 'l2', 
         'dual': False, 
         'tol':1e-4, 
         'C': 1.0,
         'fit_intercept': True, 
         'intercept_scaling': 1, 
         'class_weight': None,
         'random_state': None, 
         'solver': 'liblinear', 
         'max_iter': 100,
         'multi_class': 'ovr', 
         'verbose': 0, 
         'warm_start': False, 
         'n_jobs': 1
        },
    "ratios": {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        "test": 0.4,
        "val": 0.,
        "holdout": 0.0,
        "train": 0.6
    },
    "batch_size": {
        "train": None,
        "test": None,
        "val": None,
        "holdout": None
    }
}


ORIGINAL_LOG_FUNC = np.log

def stable_log(x):
    return ORIGINAL_LOG_FUNC(x + LPU.constants.EPSILON)

LOG = LPU.utils.utils_general.configure_logger(__name__)

# Optional dynamic import for Ray
try:
    import ray.util.client
    import ray.train
    RAY_AVAILABLE = True
except ImportError:
    LOG.warning("Ray is not available. Please install Ray to enable distributed training.")
    RAY_AVAILABLE = False

def train_model(config=None):


    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(DEFAULT_CONFIG, config)

    LPU.utils.utils_general.set_seed(LPU.constants.RANDOM_STATE)

    lpu_dataset = LPU.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')
    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    sarpu_em_model = LPU.models.SARPU.sarpu_em.SARPU(config, training_size=len(dataloaders_dict['train'].dataset))

    all_scores_dict = {split: {} for split in ['train', 'val']}
    scores_dict = {split: {} for split in ['train', 'val']}

    scores_dict['train'] = sarpu_em_model.train(dataloaders_dict['train'])
    if 'val' in dataloaders_dict:
        scores_dict['val'] = sarpu_em_model.validate(dataloaders_dict['val'], loss_fn=sarpu_em_model.loss_fn)

    for split in ['train', 'val']:
        if 'val' in dataloaders_dict and split == 'val':
            all_scores_dict[split].update(scores_dict[split])

    # Evaluate on the test set after training
    scores_dict['test'] = sarpu_em_model.validate(dataloaders_dict['test'], loss_fn=sarpu_em_model.loss_fn)

    LOG.info(f"Scores: {scores_dict}")
    # LPU.utils.plot_utils.plot_scores(all_scores_dict, loss_type='overall_loss')

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
        return all_scores_dict

if __name__ == "__main__":
    with unittest.mock.patch.object(np, 'log', stable_log) as mock_info:
        # since https://github.com/ML-KULeuven/SAR-PU/blob/c51f8af2b9604c363b6d129b7ad03b8db488346f/sarpu/sarpu/pu_learning.py#L162
        # can lead to unstable values (log of zero probs) we change the log function
        train_model()