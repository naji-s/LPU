
import copy
import sys

# import lpu.external_libs.PU_learning.train_PU

# import lpu.external_libs.potentially toched versions

sys.path.append('lpu/external_libs/PU_learning')
sys.path.append('lpu/external_libs/PU_learning/data_helper')

import lpu.external_libs.PU_learning.estimator

import lpu.external_libs.PU_learning.data_helper
import lpu.external_libs.PU_learning.helper
import lpu.external_libs.PU_learning.utils

from matplotlib import pyplot as plt
import torch.nn
import torch.backends.cudnn

import torch.utils.data
import numpy as np

import lpu.constants
import lpu.utils.dataset_utils
import lpu.datasets.LPUDataset
import lpu.external_libs.PU_learning.algorithm
# import lpu.models.mpe_model
import lpu.utils.plot_utils
import lpu.utils.utils_general
import mpe_utils
import mpe_utils_data


LOG = lpu.utils.utils_general.configure_logger(__name__)

# Optional dynamic import for Ray
try:
    import ray.util.client
    import ray.train
    RAY_AVAILABLE = True
except ImportError:
    LOG.warning("Ray is not available. Please install Ray to enable distributed training.")
    RAY_AVAILABLE = False

DEFAULT_CONFIG = {
    "dataset": "animal_no_animal",
    "lr": 0.01,
    "wd": 0.0005,
    "momentum": 0.9,
    "data_type": "mnist_17",
    "train_method": "TEDn",
    "net_type": "UCI_FCN",
    "sigmoid_loss": True,
    "estimate_alpha": True,
    "warm_start": True,
    "warm_start_epochs": 100,
    "epochs": 100,
    "alpha": 0.5,
    "beta": 0.5,
    "log_dir": "logging_accuracy",
    "data_dir": "data",
    "optimizer": "Adam",
    "alpha_estimate": 0.0,
    "show_bar": False,
    "use_alpha": False,
    "device": "cpu",
    "dataset_kind": "LPU",
    "ratios": {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        "test": 0.25,
        "val": 0.25,
        # "holdout": 0.05,
        "train": 0.5
    },
    "batch_size": {
        "train": 64,
        "test": None,
        "val": None,
        "holdout": None
    },
    "data_generating_process": "SB"  # either of CC (case-control) or SB (selection-bias)
}

def create_dataloaders_dict_mpe(config, drop_last=False):
    # dataloders_dict = {}
    # samplers_dict = {}
    device = config['device']

    mpe_dataset_dict = {}
    mpe_dataloaders_dict = {}
    mpe_indices_dict = {}
    ratios_dict = config['ratios']
    data_generating_process = config['data_generating_process']
    data_type = lpu.constants.DTYPE
    beta = config['beta']
    if config['dataset_kind'] == 'LPU':
        lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')    
        l_y_cat_transformed = lpu_dataset.l.cpu().numpy() * 2 + lpu_dataset.y.cpu().numpy()
        split_indices_dict = lpu.utils.dataset_utils.index_group_split(np.arange(len(l_y_cat_transformed)), ratios_dict=ratios_dict, random_state=lpu.constants.RANDOM_STATE, strat_arr=l_y_cat_transformed)
        for split in split_indices_dict.keys():
            # *** DO NOT DELETE *** for the normal case where we have a LPU dataset
            # samplers_dict[split], dataloders_dict[split] = lpu.utils.dataset_utils.make_data_loader(lpu_dataset, batch_size=config['batch_size'][split],)
            mpe_dataset_dict[split], mpe_indices_dict[split] = lpu.utils.dataset_utils.LPUD_to_MPED(lpu_dataset=lpu_dataset, indices=split_indices_dict[split], data_generating_process=data_generating_process)
            mpe_dataloaders_dict[split] = {}
            for dataset_type in mpe_dataset_dict[split].keys():
                mpe_dataloaders_dict[split][dataset_type] = torch.utils.data.DataLoader(mpe_dataset_dict[split][dataset_type], batch_size=config['batch_size'][split], drop_last=drop_last, shuffle=True)
            return mpe_dataloaders_dict
    elif config['dataset_kind'] == 'MPE':
        p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata, p_traindata = \
                lpu.external_libs.PU_learning.helper.get_dataset(config['data_dir'], config['data_type'], config['net_type'], device, config['alpha'], beta, config['batch_size'])


        mpe_dataloaders_dict['train']= {}
        mpe_dataloaders_dict['test'] ={}
        mpe_dataloaders_dict['val'] = {}
        mpe_dataloaders_dict['holdout'] = {}

        mpe_dataloaders_dict['train']['PDataset'] = p_trainloader
        mpe_dataloaders_dict['train']['UDataset'] = u_trainloader

        mpe_dataloaders_dict['test']['PDataset'] = p_validloader
        mpe_dataloaders_dict['test']['UDataset'] = u_validloader

        mpe_dataloaders_dict['holdout']['PDataset'] = p_validloader
        mpe_dataloaders_dict['holdout']['UDataset'] = u_validloader


        mpe_dataloaders_dict['val']['PDataset'] = p_validloader
        mpe_dataloaders_dict['val']['UDataset'] = u_validloader
        return mpe_dataloaders_dict, net
    else:
        raise ValueError("Dataset needs to be either LPU or MPE")


def train_model(config=None):
    if config is None:
        config = {}

    # Load the base configuration
    base_config = lpu.utils.utils_general.deep_update(DEFAULT_CONFIG, config)

    device = base_config['device']
    alpha = base_config['alpha']
    beta = base_config['beta']
    data_dir = base_config['data_dir']
    data_type = base_config['data_type']
    net_type = base_config['net_type']
    show_bar = base_config['show_bar']
    warm_start = base_config['warm_start']
    warm_start_epochs = base_config['warm_start_epochs']

    batch_size = base_config['batch_size']['train']
    estimate_alpha = base_config['estimate_alpha']
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)

    y_vals = []
    y_probs = []
    y_ests = []
    
    # mpe_dataloaders_dict = create_dataloaders_dict_mpe(base_config)
    (p_trainloader, u_trainloader, 
     p_testloader, u_testloader, 
     net, X, Y, p_testdata, 
     u_testdata, u_traindata) = \
        lpu.external_libs.PU_learning.helper.get_dataset(data_dir, data_type,net_type, device, alpha, beta, batch_size)

    train_unlabeled_size= len(Y)

    if device.startswith('cuda'):
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    model_parameters = net.parameters()

    if base_config['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(model_parameters, lr=base_config['lr'], momentum=base_config['momentum'], weight_decay=base_config['wd'])
    elif base_config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=base_config['lr'], weight_decay=base_config['wd'])
    elif base_config['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(model_parameters, lr=base_config['lr'])

    scores_dict = {}
    # all_scores_dict = {split: {'epochs': []} for split in mpe_dataloaders_dict.keys()}


    ## Train in the beginning for warm start
    if warm_start:
        for epoch in range(warm_start_epochs):
            train_acc = mpe_utils.train(epoch, net, p_trainloader, u_trainloader, \
                    optimizer=optimizer, criterion=criterion, device=device, show_bar=show_bar)

            # valid_acc = mpe_utils.validate(epoch, net, u_testloader, \
            #     criterion=criterion, device=device, threshold=0.5*beta/(beta + (1-beta) * alpha),show_bar=show_bar)
            
            test_acc = mpe_utils.validate(epoch, net, u_testloader, \
                criterion=criterion, device=device, threshold=0.5*beta/(beta + (1-beta) * alpha),show_bar=show_bar)
            
            if estimate_alpha: 
                pos_probs = lpu.external_libs.PU_learning.estimator.p_probs(net, device, p_trainloader)
                unlabeled_probs, unlabeled_targets = lpu.external_libs.PU_learning.estimator.u_probs(net, device, u_trainloader)


                alpha_estimate, _, _ = lpu.external_libs.PU_learning.estimator.BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
                LOG.info(f"Epoch:{epoch}, Train Acc:{train_acc}, Test Acc:{test_acc}, {alpha_estimate}")
            else: 
                LOG.info(f"Epoch:{epoch}, Train Acc:{train_acc}, Test Acc:{test_acc}")

    best_val_loss = float('inf')
    best_epoch = -1

    alpha_used = alpha_estimate
    for epoch in range(base_config['epochs']):

        if config['use_alpha']: 
            alpha_used =  alpha_estimate
        else:
            alpha_used = config['alpha']
        
        keep_samples, neg_reject = lpu.external_libs.PU_learning.algorithm.rank_inputs(epoch, net, u_trainloader, device,\
             alpha_used, u_size=train_unlabeled_size)
        train_acc = mpe_utils.train_PU_discard(epoch, net,  p_trainloader, u_trainloader,\
            optimizer, criterion, device, keep_sample=keep_samples,show_bar=show_bar)

        valid_acc = mpe_utils.validate(epoch, net, u_validloader, \
            criterion=criterion, device=device, threshold=0.5,show_bar=show_bar)
        test_acc = mpe_utils.validate(epoch, net, u_testloader, \
            criterion=criterion, device=device, threshold=0.5,show_bar=show_bar)


    
        if estimate_alpha:
            pos_probs = lpu.external_libs.PU_learning.estimator.p_probs(net, device, p_validloader)
            unlabeled_probs, unlabeled_targets = lpu.external_libs.PU_learning.estimator.u_probs(net, device, u_validloader)

            alpha_estimate, _, _ = lpu.external_libs.PU_learning.estimator.BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)

            LOG.info(f"Epoch:{epoch}, Train Acc:{train_acc}, Val Acc:{valid_acc}, Test Acc:{test_acc}, {alpha_estimate}")

        else:
            LOG.info(f"Epoch:{epoch}, Train Acc:{train_acc}, Val Acc:{valid_acc}, Test Acc:{test_acc}")
        # LOG.info(f"Train Epoch {epoch}: {scores_dict}")

        # all_scores_dict['train']['epochs'].append(epoch + base_config['warm_start_epochs'])

        # Update best validation loss and epoch
        # if scores_dict['val']['overall_loss'] < best_val_loss:
        #     best_val_loss = scores_dict['val']['overall_loss']
        #     best_epoch = epoch
        #     best_scores_dict = copy.deepcopy(scores_dict)

        # for split in mpe_dataloaders_dict.keys():
        #     for score_type, score_value in scores_dict[split].items():
        #         if score_type not in all_scores_dict[split]:
        #             all_scores_dict[split][score_type] = []
        #         all_scores_dict[split][score_type].append(score_value)


        # for split in mpe_dataloaders_dict.keys():
        #     for score_type, score_values in all_scores_dict[split].items():
        #         if score_type != 'epochs':
        #             all_scores_dict[split][score_type] = np.array(score_values)
    

        # Flatten scores_dict
        # flattened_scores = lpu.utils.utils_general.flatten_dict(scores_dict)
        # filtered_scores_dict = {}
        # for key, value in flattened_scores.items():
        #     if 'train' in key or 'val' in key or 'test' in key:
        #         if 'epochs' not in key:
        #             filtered_scores_dict[key] = value
        # print("Reporting Metrics: ", filtered_scores_dict)  # Debug print to check keys

        # # Report metrics if executed under Ray Tune
        # if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
        #     ray.train.report(filtered_scores_dict)
        # else:
        #     return all_scores_dict, best_epoch    

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    results, best_epoch = train_model()
    # lpu.utils.plot_utils.plot_scores(results, best_epoch=best_epoch)