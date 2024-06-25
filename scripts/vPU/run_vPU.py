import copy
import math
import tempfile
import os

import sys

sys.path.append('LPU/external_libs/PU_learning')
import LPU.models.MPE.MPE

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
import LPU.models.vPU.vPU

import LPU.utils.auxiliary_models
import LPU.utils.plot_utils
import LPU.utils.utils_general



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

def train_model(config=None, dataloaders_dict=None, with_ray=False):
    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(LPU.models.vPU.vPU.DEFAULT_CONFIG, config)

    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
        # setting the seed for the training
        LPU.utils.utils_general.set_seed(random_state)


    


    positive_label_list = get_positive_labels(config['dataset_name'])

    ###########################################################################
    # START: data preparation
    ###########################################################################
    if dataloaders_dict is None:
        if config['dataset_kind'] in ['LPU', 'MPE']:
            dataloaders_dict = LPU.models.MPE.MPE.create_dataloaders_dict_mpe(config)
        else:
            get_loaders = get_loaders_by_dataset_name(config['dataset_name'])
            # TODO: make sure the datasets are balanced coming out of this
            x_loader, p_loader, val_x_loader, val_p_loader,  test_loader, idx = get_loaders(batch_size=config['batch_size']['train'], num_labeled=config['num_labeled'], positive_label_list=positive_label_list)
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
    vPU_model = LPU.models.vPU.vPU.vPU(config=config, input_dim=dataloaders_dict['train']['UDataset'].dataset.data.shape[1])

    l_mean = len(dataloaders_dict['train']['PDataset'].dataset) / (len(dataloaders_dict['train']['UDataset'].dataset) + len(dataloaders_dict['train']['PDataset'].dataset))
    lr_phi = config['learning_rate']
    opt_phi = torch.optim.Adam(vPU_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    all_scores_dict = {split: {'epochs': []} for split in ['train', 'val']}
    scores_dict = {split: {} for split in ['train', 'val']}

    best_epoch = -1
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(vPU_model.state_dict())
    for epoch in range(config['epochs']):
        # adjust the optimizer
        if epoch % 20 == 19:
            lr_phi /= 2
            opt_phi = torch.optim.Adam(vPU_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

        scores_dict['train'] = vPU_model.train_one_epoch(config=config,
                                                                             opt_phi=opt_phi,
                                                                             p_loader=dataloaders_dict['train']['PDataset'],
                                                                             u_loader=dataloaders_dict['train']['UDataset'])
        # variational loss is the overall loss
        avg_var_loss = scores_dict['train']['overall_loss']
        
        avg_phi_loss = scores_dict['train']['phi_loss']
        avg_reg_loss = scores_dict['train']['reg_loss']

        all_scores_dict['train']['epochs'].append(epoch)

        if config['dataset_kind'] in ['LPU', 'MPE']:
            test_loader = None
        scores_dict['val'] = vPU_model.validate(train_p_loader=dataloaders_dict['train']['PDataset'],
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
            best_model_state = copy.deepcopy(vPU_model.state_dict())

        # Add checkpointing code
        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch,
                     "model_state": vPU_model.state_dict(),
                     "config": config,},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                ray.train.report(metrics={
                    'val_overall_loss': scores_dict['val']['overall_loss'],
                    'val_y_auc': scores_dict['val']['y_auc'],
                    'val_y_accuracy': scores_dict['val']['y_accuracy'],
                    'val_y_APS': scores_dict['val']['y_APS'],
                    'epoch': epoch,}, checkpoint=ray.train.Checkpoint.from_directory(tempdir))

    LOG.info(f"Best epoch: {best_epoch}, Best validation overall_loss: {best_val_loss:.5f}")

    model = vPU_model
    # Evaluate on the test set with the best model based on the validation set
    model.load_state_dict(best_model_state)

    # Flatten scores_dict
    flattened_scores = LPU.utils.utils_general.flatten_dict(best_scores_dict)
    filtered_scores_dict = {}
    for key, value in flattened_scores.items():
        if 'train' in key or 'val' in key:
            if 'epochs' not in key:
                filtered_scores_dict[key] = value

    # Report metrics if executed under Ray Tune
    if with_ray:
        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            ray.train.report(filtered_scores_dict)
        else:
            raise ValueError("Ray is not connected or initialized. Please connect to Ray to use Ray functionalities.")
    else:
        # Evaluate on the test set after training
        scores_dict['test'] = vPU_model.validate(train_p_loader=dataloaders_dict['train']['PDataset'],
                                                 train_u_loader=dataloaders_dict['train']['UDataset'],
                                                 val_p_loader=dataloaders_dict['test']['PDataset'],
                                                 val_u_loader=dataloaders_dict['test']['UDataset'], epoch=epoch, train_phi_loss=None,
                                                 var_loss=None, train_reg_loss=None, test_loader=test_loader)
        flattened_scores = LPU.utils.utils_general.flatten_dict(scores_dict)
        filtered_scores_dict = {}
        for key, value in flattened_scores.items():
            if 'train' in key or 'val' in key or 'test' in key:
                if 'epochs' not in key:
                    filtered_scores_dict[key] = value
        LOG.info(f"Final test scores: {scores_dict['test']}")
        return all_scores_dict, best_epoch

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    results, best_epoch = train_model()
    LPU.utils.plot_utils.plot_scores(results, loss_type='overall_loss', best_epoch=best_epoch)