import copy
import datetime
import json
import math
import os
import types
import logging

import torch

import LPU.constants
import LPU.utils.dataset_utils
import LPU.models.distPU
import LPU.utils.plot_utils
import LPU.utils.utils_general

torch.set_default_dtype(LPU.constants.DTYPE)



DEFAULT_CONFIG = {
    "device": "cpu",
    "dataset_name": "animal_no_animal",  # fmnist
    "datapath": None,  # data/FMNIST_data
    "num_labeled": None,  # 10000
    "num_workers": 1,
    "loss": "Dist-PU",
    "data_generating_process": "SS",
    "warm_up_lr": 0.001,
    "lr": 0.001,
    "warm_up_weight_decay": 0.005,
    "weight_decay": 0.001,
    "optimizer": "adam",
    "schedular": "cos-ann",
    "entropy": 1,
    "co_mu": 0.002,
    "co_entropy": 0.004,
    "alpha": 6.0,
    "co_mix_entropy": 0.04,
    "co_mixup": 5.0,
    "warm_up_epochs": 10,
    "pu_epochs": 10,
    "random_seed": 0,
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

    "dataset_kind": "LPU",
    "batch_size": 
    {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    },
    "best_model_loc": "/Users/naji/phd_codebase/LPU/scripts/distPU/best_model_checkpoints"
}


LOG = LPU.utils.utils_general.configure_logger(__name__)

# Optional dynamic import for Ray
try:
    import ray.util.client
    import ray.train
    RAY_AVAILABLE = True
except ImportError:
    LOG.warning("Ray is not available. Please install Ray to enable distributed training.")
    RAY_AVAILABLE = False

def update_co_entropy(config, epoch):
    co_entropy = (1 - math.cos((float(epoch) / config['pu_epochs']) * (math.pi / 2))) * config['co_entropy']
    return co_entropy

def train_model(config=None):

    if config is None:
        config = {}

    # Load the base configuration
    base_config = LPU.utils.utils_general.deep_update(DEFAULT_CONFIG, config)

    LPU.utils.utils_general.set_seed(LPU.constants.RANDOM_STATE)

    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(base_config, transform=None, target_transform=None)

    if base_config['dataset_kind'] == 'LPU':
        dim = dataloaders_dict['train'].dataset.X.shape[1]
    elif base_config['dataset_kind'] == 'distPU':
        dim = torch.flatten(dataloaders_dict['train'].dataset.X, 1).shape[1]

    distPU_model = LPU.models.distPU.distPU(base_config, dim)
    distPU_model.set_C(dataloaders_dict['train'])

    loss_fn = LPU.models.distPU.create_loss(base_config, prior=distPU_model.prior)
    warm_up_lr = base_config['warm_up_lr']
    warm_up_weight_decay = base_config['warm_up_weight_decay']
    warm_up_epochs = base_config['warm_up_epochs']
    optimizer = torch.optim.Adam(
        distPU_model.parameters(), lr=warm_up_lr, weight_decay=warm_up_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, warm_up_epochs)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}

    best_val_loss = float('inf')
    best_epoch = -1
    best_scores_dict = None
    best_model_state = None

    # Warmup
    for epoch in range(warm_up_epochs):
        scores_dict = {split: {} for split in dataloaders_dict.keys()}
        scores_dict_item = distPU_model.train_one_epoch(epoch=epoch, dataloader=dataloaders_dict['train'], loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
        scores_dict['train'].update(scores_dict_item)

        for split in ['train', 'val']:
            scores_dict_item = distPU_model.validate(dataloaders_dict[split], loss_fn=loss_fn, model=distPU_model.model)
            scores_dict[split].update(scores_dict_item)
            all_scores_dict[split]['epochs'].append(epoch)

        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Warmup Epoch {epoch}: {scores_dict}")

        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            best_scores_dict = copy.deepcopy(scores_dict)
            best_model_state = copy.deepcopy(distPU_model.state_dict())

    # Training
    mixup_dataset = LPU.models.distPU.MixupDataset()
    mixup_dataset.update_psudos(dataloaders_dict['train'], distPU_model.model, distPU_model.device)

    lr = base_config['lr']
    weight_decay = base_config['weight_decay']
    pu_epochs = base_config['pu_epochs']

    for epoch in range(pu_epochs):
        co_entropy = update_co_entropy(base_config, epoch)
        LOG.info(f"Updating co-entropy: {co_entropy:.5f}")

        LOG.info("Training with mixup")
        scores_dict_item = distPU_model.train_mixup_one_epoch(epoch=warm_up_epochs+epoch, dataloader=dataloaders_dict['train'], loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, mixup_dataset=mixup_dataset, co_entropy=co_entropy)
        scores_dict['train'].update(scores_dict_item)

        for split in ['train', 'val']:
            scores_dict_item = distPU_model.validate(dataloaders_dict[split], loss_fn=loss_fn, model=distPU_model.model)
            scores_dict[split].update(scores_dict_item)
            all_scores_dict[split]['epochs'].append(warm_up_epochs+epoch)

        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)
                best_model_state = copy.deepcopy(distPU_model.state_dict())

        LOG.info(f"Training Epoch {epoch+warm_up_epochs} (counting warm up epochs): {scores_dict}")

        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = warm_up_epochs + epoch
            best_scores_dict = copy.deepcopy(scores_dict)
            best_model_state = copy.deepcopy(distPU_model.state_dict())


    LOG.info(f"Best epoch: {best_epoch}, Best validation overall_loss: {best_val_loss:.5f}")
        # Update best validation loss and epoch
    
    model = distPU_model
    # Evaluate on the test set with the best model based on the validation set
    model.load_state_dict(best_model_state)

    best_scores_dict['test'] = model.validate(dataloaders_dict['test'], loss_fn=loss_fn, model=model.model)

    # Flatten scores_dict
    flattened_scores = LPU.utils.utils_general.flatten_dict(best_scores_dict)
    filtered_scores_dict = {}
    for key, value in flattened_scores.items():
        if 'train' in key or 'val' in key or 'test' in key:
            if 'epochs' not in key:
                filtered_scores_dict[key] = value
    LOG.info(f"Final test error: {best_scores_dict['test']}")

    # Report metrics if executed under Ray Tune
    if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
        ray.train.report(filtered_scores_dict)
    else:
        return all_scores_dict, best_epoch

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    results, best_epoch = train_model()
    LPU.utils.plot_utils.plot_scores(results, best_epoch=best_epoch, loss_type='overall_loss')
