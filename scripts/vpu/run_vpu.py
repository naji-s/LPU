
import copy
import math
import sys


sys.path.append('lpu/external_libs/vpu')

import lpu.external_libs.vpu.model.model_fashionmnist
import lpu.models
import lpu.models.vpu



import lpu.external_libs.vpu.data
import lpu.external_libs.vpu.vpu


from matplotlib import pyplot as plt
import torch.nn
import torch.utils.data
import numpy as np

import lpu.constants
import lpu.datasets.dataset_utils
import lpu.datasets.LPUDataset
import lpu.external_libs
import lpu.external_libs.vpu
import lpu.external_libs.vpu.dataset
import lpu.external_libs.vpu.dataset.dataset_avila
import lpu.external_libs.vpu.dataset.dataset_cifar
import lpu.external_libs.vpu.dataset.dataset_fashionmnist
import lpu.external_libs.vpu.dataset.dataset_grid
import lpu.external_libs.vpu.dataset.dataset_pageblocks
import lpu.external_libs.vpu.dataset.dataset_stl
import lpu.external_libs.vpu.model.model_cifar
import lpu.external_libs.vpu.model.model_vec
import lpu.external_libs.vpu.model.model_stl
import lpu.external_libs.vpu.vpu
import lpu.utils.auxiliary_models
import lpu.utils.utils_general

sys.path.append('lpu/external_libs/PU_learning')
# sys.path.append('lpu/external_libs/PU_learning/utils')

import lpu.models.mpe_model
import lpu.scripts.mpe.run_mpe



    
LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
BATCH_SIZE = 32
HOLDOUT_RATIO = 0.05
VAL_RATIO = 0.10
TEST_RATIO = 0.35

def get_loaders_by_dataset_name(dataset_name):
    """ Returns the data loader functions based on the dataset name. """
    dataset_to_loader = {
        'cifar10': lpu.external_libs.vpu.dataset.dataset_cifar.get_cifar10_loaders,
        'fashionMNIST': lpu.external_libs.vpu.dataset.dataset_fashionmnist.get_fashionMNIST_loaders,
        'stl10': lpu.external_libs.vpu.dataset.dataset_stl.get_stl10_loaders,
        'pageblocks': lpu.external_libs.vpu.dataset.dataset_pageblocks.get_pageblocks_loaders,
        'grid': lpu.external_libs.vpu.dataset.dataset_grid.get_grid_loaders,
        'avila': lpu.external_libs.vpu.dataset.dataset_avila.get_avila_loaders
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


def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/vpu_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    get_loaders = get_loaders_by_dataset_name(config['dataset'])
    positive_label_list = get_positive_labels(config['dataset'])

    ###########################################################################
    # START: data preparation
    ###########################################################################
    if config['dataset_kind'] in ['LPU', 'MPE']:
        dataloaders_dict = lpu.scripts.mpe.run_mpe.create_dataloaders_dict_mpe(config)
    else:
        # TODO: make sure the datasets are balanced coming out of this
        p_loader, x_loader, val_p_loader, val_x_loader, test_loader, _ = get_loaders(batch_size=config['batch_size']['train'], num_labeled=config['num_labeled'], positive_label_list=positive_label_list)
        dataloaders_dict = {}
        dataloaders_dict['train'] = {}
        dataloaders_dict['train']['PDataset'] = p_loader
        dataloaders_dict['train']['UDataset'] = x_loader
        dataloaders_dict['val'] = {}
        dataloaders_dict['val']['PDataset'] = p_loader
        dataloaders_dict['val']['UDataset'] = x_loader

    # please read the following information to make sure it is running with the desired setting
    print('==> Preparing data')
    print('    # train data: ', len(x_loader.dataset))
    print('    # labeled train data: ', len(p_loader.dataset))
    print('    # test data: ', len(test_loader.dataset))
    print('    # val x data:', len(val_x_loader.dataset))
    print('    # val p data:', len(val_p_loader.dataset))
    

    ###########################################################################
    # START: training
    ###########################################################################
    # lpu.external_libs.vpu.vpu.run_vpu(config, loaders, lpu.external_libs.vpu.model.model_cifar.NetworkPhi)    
    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1 # highest test accuracy on test set

    vpu_model = lpu.models.vpu.VPU(config=config)

    # set up the optimizer
    lr_phi = config['learning_rate']
    opt_phi = torch.optim.Adam(vpu_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    for epoch in range(config['epochs']):

        # adjust the optimizer
        if epoch % 20 == 19:
            lr_phi /= 2
            opt_phi = torch.optim.Adam(vpu_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))
        print (epoch)
        vpu_model.train_one_epoch(config=config, opt_phi=opt_phi, p_loader=dataloaders_dict['train']['PDataset'], x_loader=dataloaders_dict['train']['UDataset'])





if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    main()

