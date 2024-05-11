
import copy
import math
import sys




sys.path.append('lpu/external_libs/PU_learning')
import lpu.scripts.mpe.run_mpe

sys.path.append('lpu/external_libs/vpu')

import lpu.external_libs.vpu.data


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
import lpu.external_libs.vpu.model.model_fashionmnist
import lpu.external_libs.vpu.model.model_stl
import lpu.external_libs.vpu.model.model_vec
import lpu.models
import lpu.models.vpu

import lpu.utils.auxiliary_models
import lpu.utils.utils_general




    
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
    positive_label_list = get_positive_labels(config['dataset_name'])

    ###########################################################################
    # START: data preparation
    ###########################################################################
    if config['dataset_kind'] in ['LPU', 'MPE']:
        dataloaders_dict = lpu.scripts.mpe.run_mpe.create_dataloaders_dict_mpe(config, drop_last=True)
        # test_labeled_X = dataloaders_dict['test']['PDataset'].dataset.data
        # test_pos_X = dataloaders_dict['test']['UDataset'].dataset.pos_data
        # test_neg_X = dataloaders_dict['test']['UDataset'].dataset.neg_data
        # test_X = np.concatenate([test_labeled_X, test_pos_X, test_neg_X], axis=0)
        # test_y = np.concatenate([np.ones(len(test_labeled_X)), np.ones(len(test_pos_X)), np.zeros(len(test_neg_X))], axis=0)
        # test_l = np.concatenate([np.ones(len(test_labeled_X)), np.zeros(len(test_pos_X)), np.zeros(len(test_neg_X))], axis=0)
        # test_loader = lpu.datasets.dataset_utils.make_data_loader(lpu.datasets.LPUDataset.LPUDataset(device=config['device'], data_dict={'X': test_X, 'l': test_l, 'y': test_y}), batch_size=config['batch_size']['test'], drop_last=False)[1]
        
    else:
        get_loaders = get_loaders_by_dataset_name(config['dataset_name'])
        # TODO: make sure the datasets are balanced coming out of this
        x_loader, p_loader, val_x_loader, val_p_loader,  test_loader, idx = get_loaders(batch_size=config['batch_size']['train'], num_labeled=config['num_labeled'], positive_label_list=positive_label_list)
        # x_loader, p_loader, val_x_loader, val_p_loader,  test_loader, idx
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
    # lpu.external_libs.vpu.vpu.run_vpu(config, loaders, lpu.external_libs.vpu.model.model_cifar.NetworkPhi)    
    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1 # highest test accuracy on test set
    vpu_model = lpu.models.vpu.VPU(config=config, input_dim=dataloaders_dict['train']['UDataset'].dataset.data.shape[1])
        
    l_mean = len(dataloaders_dict['train']['PDataset'].dataset) / (len(dataloaders_dict['train']['UDataset'].dataset) + len(dataloaders_dict['train']['PDataset'].dataset))
    # set up the optimizer
    lr_phi = config['learning_rate']
    opt_phi = torch.optim.Adam(vpu_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    for epoch in range(config['epochs']):

        # adjust the optimizer
        if epoch % 20 == 19:
            lr_phi /= 2
            opt_phi = torch.optim.Adam(vpu_model.parameters(), lr=lr_phi, betas=(0.5, 0.99))
        avg_phi_loss, avg_var_loss, avg_reg_loss = vpu_model.train_one_epoch(config=config, 
                                                                             opt_phi=opt_phi, 
                                                                             p_loader=dataloaders_dict['train']['PDataset'], 
                                                                             u_loader=dataloaders_dict['train']['UDataset'])
        # since the way original repo of VPU calcuylates error is based on the actual test set
        # with target values with no access to labeled/unlabeled values (i.e. no $l$, and only $y$)
        if config['dataset_kind'] in ['LPU', 'MPE']:
            test_loader = None
        # vpu_model.set_C(l_mean=l_mean, p_loader = dataloaders_dict['train']['PDataset'], u_loader=dataloaders_dict['train']['UDataset'])
        print(vpu_model.validate(train_p_loader=dataloaders_dict['train']['PDataset'], 
                                 train_u_loader=dataloaders_dict['train']['UDataset'], 
                                 val_p_loader=dataloaders_dict['val']['PDataset'], 
                                 val_u_loader=dataloaders_dict['val']['UDataset'], epoch=epoch, phi_loss=avg_phi_loss, 
                                 var_loss=avg_var_loss, reg_loss=avg_reg_loss, test_loader=test_loader))


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    main()

