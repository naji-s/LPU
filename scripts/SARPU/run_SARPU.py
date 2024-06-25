import logging
import unittest.mock
import tempfile
import os

import numpy as np
import torch
import LPU.constants
import LPU.datasets.LPUDataset
import sys
sys.path.append('LPU/external_libs/SAR_PU')
sys.path.append('LPU/external_libs/SAR_PU/sarpu')
sys.path.append('LPU/external_libs/SAR_PU/sarpu/sarpu')
import LPU.models.SARPU.SARPU
import LPU.utils.utils_general
import LPU.utils.dataset_utils
import LPU.utils.plot_utils
import LPU.external_libs.SAR_PU.sarpu.sarpu.PUmodels
import LPU.external_libs.SAR_PU.sarpu.sarpu.pu_learning
import LPU.external_libs.SAR_PU.sarpu
import LPU.external_libs.SAR_PU.sarpu.sarpu

torch.set_default_dtype(LPU.constants.DTYPE)

USE_DEFAULT_CONFIG = False



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

def train_model(config=None, dataloaders_dict=None, with_ray=False):
    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(LPU.models.SARPU.SARPU.DEFAULT_CONFIG, config)

    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
        # setting the seed for the training
        LPU.utils.utils_general.set_seed(random_state)



    if dataloaders_dict is None:
        dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    



    SARPU_model = LPU.models.SARPU.SARPU.SARPU(config, training_size=len(dataloaders_dict['train'].dataset))

    all_scores_dict = {split: {} for split in ['train', 'val']}
    scores_dict = {split: {} for split in ['train', 'val']}

    scores_dict['train'] = SARPU_model.train(dataloaders_dict['train'])
    if 'val' in dataloaders_dict:
        scores_dict['val'] = SARPU_model.validate(dataloaders_dict['val'], loss_fn=SARPU_model.loss_fn)

    for split in ['train', 'val']:
        if 'val' in dataloaders_dict and split == 'val':
            all_scores_dict[split].update(scores_dict[split])

    # Flatten scores_dict
    flattened_scores = LPU.utils.utils_general.flatten_dict(scores_dict)
    filtered_scores_dict = {}
    for key, value in flattened_scores.items():
        if 'train' in key or 'val' in key:
            if 'epochs' not in key:
                filtered_scores_dict[key] = value

    # Add checkpointing code
    if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {"model_state": SARPU_model.state_dict(),
                 "config": config,},
                os.path.join(tempdir, "checkpoint.pt"),
            )
            ray.train.report(metrics=filtered_scores_dict, checkpoint=ray.train.Checkpoint.from_directory(tempdir))

    # Report metrics if executed under Ray Tune
    if with_ray:
        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            ray.train.report(filtered_scores_dict)
        else:
            raise ValueError("Ray is not connected or initialized. Please connect to Ray to use Ray functionalities.")
    else:
        # Evaluate on the test set after training
        scores_dict['test'] = SARPU_model.validate(dataloaders_dict['test'], loss_fn=SARPU_model.loss_fn)
        flattened_scores = LPU.utils.utils_general.flatten_dict(scores_dict)
        filtered_scores_dict = {}
        for key, value in flattened_scores.items():
            if 'train' in key or 'val' in key or 'test' in key:
                if 'epochs' not in key:
                    filtered_scores_dict[key] = value
        LOG.info(f"Final test scores: {scores_dict['test']}")
        return all_scores_dict

if __name__ == "__main__":
    with unittest.mock.patch.object(np, 'log', stable_log) as mock_info:
        # since https://github.com/ML-KULeuven/SAR-PU/blob/c51f8af2b9604c363b6d129b7ad03b8db488346f/sarpu/sarpu/pu_learning.py#L162
        # can lead to unstable values (log of zero probs) we change the log function
        train_model()