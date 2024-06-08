import copy
import math

import sys

sys.path.append('LPU/external_libs/PU_learning')
import LPU.scripts.mpe.run_mpe

sys.path.append('LPU/external_libs/vpu')
import LPU.external_libs
import LPU.external_libs.vpu
import LPU.external_libs.vpu.dataset
import LPU.external_libs.vpu.dataset.dataset_avila
import LPU.external_libs.vpu.dataset.dataset_cifar
import LPU.external_libs.vpu.dataset.dataset_fashionmnist
import LPU.external_libs.vpu.dataset.dataset_grid
import LPU.external_libs.vpu.dataset.dataset_pageblocks
import LPU.external_libs.vpu.dataset.dataset_stl
import LPU.external_libs.vpu.model.model_cifar
import LPU.external_libs.vpu.model.model_fashionmnist
import LPU.external_libs.vpu.model.model_stl
import LPU.external_libs.vpu.model.model_vec
import LPU.external_libs.vpu.data    

from matplotlib import pyplot as plt
import torch.nn
import torch.utils.data
import numpy as np

import LPU.constants
import LPU.utils.dataset_utils
import LPU.datasets.LPUDataset

import LPU.models
import LPU.models.vpu

import LPU.utils.auxiliary_models
import LPU.utils.plot_utils
import LPU.utils.utils_general

DEFAULT_CONFIG = {
    "device": "cpu",
    "dataset_name": "animal_no_animal",  # could also be fashionMNIST
    "dataset_kind": "LPU",
    "gpu": 9,
    "val_iterations": 30,
    "num_labeled": 3000,
    "learning_rate": 0.0001,
    "epochs": 50,
    "mix_alpha": 0.3,
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "lam": 0.3,
    "random_seed": 0,
    "ratios": {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        "test": 0.25,
        "val": 0.1,
        "holdout": 0.0,
        "train": 0.65
    },
    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    }
}


def get_loaders_by_dataset_name(dataset_name):
    """ Returns the data loader functions based on the dataset name. """
    dataset_to_loader = {
        'cifar10': LPU.external_libs.vpu.dataset.dataset_cifar.get_cifar10_loaders,
        'fashionMNIST': LPU.external_libs.vpu.dataset.dataset_fashionmnist.get_fashionMNIST_loaders,
        'stl10': LPU.external_libs.vpu.dataset.dataset_stl.get_stl10_loaders,
        'pageblocks': LPU.external_libs.vpu.dataset.dataset_pageblocks.get_pageblocks_loaders,
        'grid': LPU.external_libs.vpu.dataset.dataset_grid.get_grid_loaders,
        'avila': LPU.external_libs.vpu.dataset.dataset_avila.get_avila_loaders
    }
    return dataset_to_loader.get(dataset_name)


def get_positive_labels(dataset_name):
    """ Retrieves the list of positive labels from config with a default value based on the dataset. """
    default_labels = {
        'cifar10': [0, 1, 8, 9],
        'fashionMNIST': [1, 4, 7],
        'stl10': [0, 2, 3, 8, 9],
        'pageblocks': [2, 3, 4, 5],
        'grid': [1],
        'avila': ['A']
    }
    return default_labels.get(dataset_name)

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
    base_config = LPU.utils.utils_general.deep_update(DEFAULT_CONFIG, config)

    LPU.utils.utils_general.set_seed(LPU.constants.RANDOM_STATE)

    positive_label_list = get_positive_labels(base_config['dataset_name'])

    ###########################################################################
    # START: data preparation
    ###########################################################################
    if base_config['dataset_kind'] in ['LPU', 'MPE']:
        dataloaders_dict = LPU.scripts.mpe.run_mpe.create_dataloaders_dict_mpe(base_config, drop_last=True)
    else:
        get_loaders = get_loaders_by_dataset_name(base_config['dataset_name'])
        # TODO: make sure the datasets are balanced coming out of this
        x_loader, p_loader, val_x_loader, val_p_loader,  test_loader, idx = get_loaders(batch_size=base_config['batch_size']['train'], num_labeled=base_config['num_labeled'], positive_label_list=positive_label_list)
        dataloaders_dict = {}
        dataloaders_dict['train'] = {}
        dataloaders_dict['train']['PDataset'] = p_loader
        dataloaders_dict['train']['UDataset'] = x_loader
        dataloaders_dict['val'] = {}
        dataloaders_dict['val']['PDataset'] = val_p_loader
        dataloaders_dict['val']['UDataset'] = val_x_loader

    ###########################################################################
    # START: training
    ###########################################################################
    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1 # highest test accuracy on test set
    vpu_model = LPU.models.vpu.VPU(config=base_config, input_dim=dataloaders_dict['train']['UDataset'].dataset.data.shape[1])

    l_mean = len(dataloaders_dict['train']['PDataset'].dataset) / (len(dataloaders_dict['train']['UDataset'].dataset) + len(dataloaders_dict['train']['PDataset'].dataset))
    lr_phi = base_config['learning_rate']
    opt_phi = torch.optim.Adam(vpu_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    all_scores_dict = {split: {'epochs': []} for split in ['train', 'val']}
    scores_dict = {split: {} for split in ['train', 'val']}

    best_epoch = -1
    best_val_loss = float('inf')
    for epoch in range(base_config['epochs']):
        # adjust the optimizer
        if epoch % 20 == 19:
            lr_phi /= 2
            opt_phi = torch.optim.Adam(vpu_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

        scores_dict['train'] = vpu_model.train_one_epoch(config=base_config,
                                                                             opt_phi=opt_phi,
                                                                             p_loader=dataloaders_dict['train']['PDataset'],
                                                                             u_loader=dataloaders_dict['train']['UDataset'])
        # variational loss is the overall loss
        avg_var_loss = scores_dict['train']['overall_loss']
        
        avg_phi_loss = scores_dict['train']['phi_loss']
        avg_reg_loss = scores_dict['train']['reg_loss']

        all_scores_dict['train']['epochs'].append(epoch)

        if base_config['dataset_kind'] in ['LPU', 'MPE']:
            test_loader = None
        scores_dict['val'] = vpu_model.validate(train_p_loader=dataloaders_dict['train']['PDataset'],
                                                train_u_loader=dataloaders_dict['train']['UDataset'],
                                                val_p_loader=dataloaders_dict['val']['PDataset'],
                                                val_u_loader=dataloaders_dict['val']['UDataset'], epoch=epoch, train_phi_loss=avg_phi_loss,
                                                var_loss=avg_var_loss, train_reg_loss=avg_reg_loss, test_loader=test_loader)
        all_scores_dict['val']['epochs'].append(epoch)

        for split in ['train', 'val']:
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Epoch {epoch}: {scores_dict}")
        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            best_scores_dict = copy.deepcopy(scores_dict)

    # Evaluate on the test set after training
    scores_dict['test'] = vpu_model.validate(train_p_loader=dataloaders_dict['train']['PDataset'],
                                             train_u_loader=dataloaders_dict['train']['UDataset'],
                                             val_p_loader=dataloaders_dict['test']['PDataset'],
                                             val_u_loader=dataloaders_dict['test']['UDataset'], epoch=epoch, train_phi_loss=None,
                                             var_loss=None, train_reg_loss=None, test_loader=test_loader)


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
        return all_scores_dict, best_epoch

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    results, best_epoch = train_model()
    LPU.utils.plot_utils.plot_scores(results, loss_type='overall_loss', best_epoch=best_epoch)
