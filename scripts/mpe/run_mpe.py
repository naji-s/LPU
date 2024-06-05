
import copy


import sys
sys.path.append('LPU/external_libs/PU_learning')
sys.path.append('LPU/external_libs/PU_learning/data_helper')

import LPU.external_libs.PU_learning.helper
import LPU.external_libs.PU_learning.utils
import LPU.external_libs.PU_learning.algorithm

from matplotlib import pyplot as plt
import torch.nn
import torch.utils.data
import numpy as np

import LPU.constants
import LPU.utils.dataset_utils
import LPU.datasets.LPUDataset
import LPU.models.mpe_model
import LPU.utils.plot_utils
import LPU.utils.utils_general    


LOG = LPU.utils.utils_general.configure_logger(__name__)

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
    "data_type": None,
    "train_method": "TEDn",
    "net_type": "LeNet",
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
        "val": 0.2,
        "holdout": 0.05,
        "train": 0.5
    },
    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    },
    "data_generating_process": "SB"  # either of CC (case-control) or SB (selection-bias)
}

def create_dataloaders_dict_mpe(config, drop_last=False):
    # dataloders_dict = {}
    # samplers_dict = {}
    mpe_dataset_dict = {}
    mpe_dataloaders_dict = {}
    mpe_indices_dict = {}
    ratios_dict = config['ratios']
    data_generating_process = config['data_generating_process']
    data_type = LPU.constants.DTYPE
    if config['dataset_kind'] == 'LPU':
        lpu_dataset = LPU.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')    
        l_y_cat_transformed = lpu_dataset.l.cpu().numpy() * 2 + lpu_dataset.y.cpu().numpy()
        split_indices_dict = LPU.utils.dataset_utils.index_group_split(np.arange(len(l_y_cat_transformed)), ratios_dict=ratios_dict, random_state=LPU.constants.RANDOM_STATE, strat_arr=l_y_cat_transformed)
        for split in split_indices_dict.keys():
            # *** DO NOT DELETE *** for the normal case where we have a LPU dataset
            # samplers_dict[split], dataloders_dict[split] = LPU.utils.dataset_utils.make_data_loader(lpu_dataset, batch_size=config['batch_size'][split],)
            mpe_dataset_dict[split], mpe_indices_dict[split] = LPU.utils.dataset_utils.LPUD_to_MPED(lpu_dataset=lpu_dataset, indices=split_indices_dict[split], data_generating_process=data_generating_process)
            mpe_dataloaders_dict[split] = {}
            for dataset_type in mpe_dataset_dict[split].keys():
                mpe_dataloaders_dict[split][dataset_type] = torch.utils.data.DataLoader(mpe_dataset_dict[split][dataset_type], batch_size=config['batch_size'][split], drop_last=drop_last, shuffle=True)
    elif config['dataset_kind'] == 'MPE':
        p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata, p_traindata = \
                LPU.external_libs.PU_learning.helper.get_dataset(config['data_dir'], config['data_type'], config['net_type'], config['device'], config['alpha'], config['beta'], config['batch_size'])


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
    else:
        raise ValueError("Dataset needs to be either LPU or MPE")
    return mpe_dataloaders_dict


def train_model(config=None):

    if config is None:
        config = {}
    # Load the base configuration
    base_config = LPU.utils.utils_general.deep_update(DEFAULT_CONFIG, config)

    LPU.utils.utils_general.set_seed(LPU.constants.RANDOM_STATE)

    criterion = torch.nn.CrossEntropyLoss()

    mpe_model = LPU.models.mpe_model.MPE(base_config)

    mpe_dataloaders_dict = create_dataloaders_dict_mpe(base_config)

    mpe_model.initialize_model(mpe_dataloaders_dict['train']['UDataset'].dataset.data.shape[1])
    train_unlabeled_size = len(mpe_dataloaders_dict['train']['UDataset'].dataset.data)
 
    if base_config['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(mpe_model.net.parameters(), lr=base_config['lr'], momentum=base_config['momentum'], weight_decay=base_config['wd'])
    elif base_config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(mpe_model.net.parameters(), lr=base_config['lr'], weight_decay=base_config['wd'])
    elif base_config['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(mpe_model.net.parameters(), lr=base_config['lr'])

    scores_dict = {}
    all_scores_dict = {split: {'epochs': []} for split in mpe_dataloaders_dict.keys()}

    ## Train in the beginning for warm start
    if base_config['warm_start']:
        for epoch in range(base_config['warm_start_epochs']):
            scores_dict['train'] = mpe_model.warm_up_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'],
                                                       u_trainloader=mpe_dataloaders_dict['train']['UDataset'],
                                                       optimizer=optimizer, criterion=criterion, valid_loader=None)
            all_scores_dict['train']['epochs'].append(epoch)

            scores_dict['val'] = mpe_model.validate(epoch, p_validloader=mpe_dataloaders_dict['val']['PDataset'],
                                                u_validloader=mpe_dataloaders_dict['val']['UDataset'],
                                                criterion=criterion, threshold=0.5)
            all_scores_dict['val']['epochs'].append(epoch)

            if base_config['estimate_alpha']:
                mpe_model.alpha_estimate = mpe_model.estimate_alpha(p_holdoutloader=mpe_dataloaders_dict['holdout']['PDataset'],
                                                                    u_holdoutloader=mpe_dataloaders_dict['holdout']['UDataset'])
                mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))
            LOG.info(f"Warmup Epoch {epoch}: {scores_dict}")

            for split in ['train', 'val']:
                for score_type, score_value in scores_dict[split].items():
                    if score_type not in all_scores_dict[split]:
                        all_scores_dict[split][score_type] = []
                    all_scores_dict[split][score_type].append(score_value)

    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(base_config['epochs']):
        train_scores = mpe_model.train_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'],
                                                 u_trainloader=mpe_dataloaders_dict['train']['UDataset'],
                                                 optimizer=optimizer, criterion=criterion,
                                                 train_unlabeled_size=train_unlabeled_size)
        scores_dict['train'] = train_scores
        all_scores_dict['train']['epochs'].append(epoch + base_config['warm_start_epochs'])

        scores_dict['val'] = mpe_model.validate(epoch, p_validloader=mpe_dataloaders_dict['val']['PDataset'],
                                                u_validloader=mpe_dataloaders_dict['val']['UDataset'],
                                                criterion=criterion, threshold=0.5)
        all_scores_dict['val']['epochs'].append(epoch + base_config['warm_start_epochs'])

        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            best_scores_dict = copy.deepcopy(scores_dict)

        for split in ['train', 'val']:
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        if base_config['estimate_alpha']:
            mpe_model.alpha_estimate = mpe_model.estimate_alpha(mpe_dataloaders_dict['holdout']['PDataset'], mpe_dataloaders_dict['holdout']['UDataset'])
            mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))

        LOG.info(f"Train Epoch {epoch}: {scores_dict}")



    for split in mpe_dataloaders_dict.keys():
        for score_type, score_values in all_scores_dict[split].items():
            if score_type != 'epochs':
                all_scores_dict[split][score_type] = np.array(score_values)


    scores_dict['test'] = mpe_model.validate(epoch=0, p_validloader=mpe_dataloaders_dict['test']['PDataset'],
                                             u_validloader=mpe_dataloaders_dict['test']['UDataset'],
                                             criterion=criterion, threshold=0.5)


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
    LPU.utils.plot_utils.plot_scores(results, best_epoch=best_epoch)
