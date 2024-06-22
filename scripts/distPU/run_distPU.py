import copy
import datetime
import json
import math
import os
import types
import logging
import tempfile

import torch

import LPU.constants
import LPU.models.distPU.distPU
import LPU.utils.dataset_utils
import LPU.models.distPU
import LPU.utils.plot_utils
import LPU.utils.utils_general

torch.set_default_dtype(LPU.constants.DTYPE)


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

def train_model(config=None, dataloaders_dict=None, with_ray=False):

    if config is None:
        config = {}

    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(LPU.models.distPU.distPU.DEFAULT_CONFIG, config)

    LPU.utils.utils_general.set_seed(LPU.constants.RANDOM_STATE)

    if dataloaders_dict is None:
        dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config, transform=None, target_transform=None)

    if config['dataset_kind'] == 'LPU':
        dim = dataloaders_dict['train'].dataset.X.shape[1]
    elif config['dataset_kind'] == 'distPU':
        dim = torch.flatten(dataloaders_dict['train'].dataset.X, 1).shape[1]

    distPU_model = LPU.models.distPU.distPU.distPU(config, dim)
    distPU_model.set_C(dataloaders_dict['train'])

    loss_fn = LPU.models.distPU.distPU.create_loss(config, prior=distPU_model.prior)
    warm_up_lr = config['warm_up_lr']
    warm_up_weight_decay = config['warm_up_weight_decay']
    warm_up_epochs = config['warm_up_epochs']
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

        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, 
                     "model_state": distPU_model.state_dict(),
                     "config": config,},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                ray.train.report(metrics={
                        'val_overall_loss': scores_dict['val']['overall_loss'],
                        'val_y_auc': scores_dict['val']['y_auc'],
                        'val_y_accuracy': scores_dict['val']['y_accuracy'],
                        'val_y_APS': scores_dict['val']['y_APS'],
                        'epoch': epoch,}, checkpoint=ray.train.Checkpoint.from_directory(tempdir))

    # Training
    mixup_dataset = LPU.models.distPU.distPU.MixupDataset()
    mixup_dataset.update_psudos(dataloaders_dict['train'], distPU_model.model, distPU_model.device)

    lr = config['lr']
    weight_decay = config['weight_decay']
    pu_epochs = config['pu_epochs']

    for epoch in range(pu_epochs):
        co_entropy = update_co_entropy(config, epoch)
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

        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, 
                     "model_state": distPU_model.state_dict(),
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
        # Update best validation loss and epoch
    
    model = distPU_model
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
        best_scores_dict['test'] = model.validate(dataloaders_dict['test'], loss_fn=loss_fn, model=model.model)
        flattened_scores = LPU.utils.utils_general.flatten_dict(best_scores_dict)
        filtered_scores_dict = {}
        for key, value in flattened_scores.items():
            if 'train' in key or 'val' in key or 'test' in key:
                if 'epochs' not in key:
                    filtered_scores_dict[key] = value
        LOG.info(f"Final test scores: {best_scores_dict['test']}")
        return all_scores_dict, best_epoch

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    results, best_epoch = train_model()
    LPU.utils.plot_utils.plot_scores(results, best_epoch=best_epoch, loss_type='overall_loss')