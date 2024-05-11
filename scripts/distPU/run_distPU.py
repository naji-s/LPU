import copy
import math
import sys
import types
import logging

sys.path.append('lpu/external_libs/distPU')
sys.path.append('lpu/external_libs/distPU/dataTools')

import torch
import lpu.constants
import lpu.datasets.dataset_utils
import lpu.models.distPU
import lpu.utils.plot_utils
import lpu.utils.utils_general

torch.set_default_dtype(lpu.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    'warm_up_lr': 0.01,
    'warm_up_weight_decay': 0.0,
    'warm_up_epochs': 10,
    'lr': 0.01,
    'weight_decay': 0.0,
    'pu_epochs': 50,
    'co_entropy': 0.1
}

LOG = lpu.utils.utils_general.configure_logger(__name__)

def update_co_entropy(config, epoch):
    co_entropy = (1 - math.cos((float(epoch) / config['pu_epochs']) * (math.pi / 2))) * config['co_entropy']
    return co_entropy

def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/distPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    args = types.SimpleNamespace(**config)

    dataloaders_dict = lpu.datasets.dataset_utils.create_dataloaders_dict(config, transform=None, target_transform=None)

    if config['dataset_kind'] == 'LPU':
        dim = dataloaders_dict['train'].dataset.X.shape[1]
    elif config['dataset_kind'] == 'distPU':
        dim = torch.flatten(dataloaders_dict['train'].dataset.X, 1).shape[1]

    distPU_model = lpu.models.distPU.distPU(config, dim)
    distPU_model.set_C(copy.deepcopy(dataloaders_dict['train']))

    loss_fn = lpu.models.distPU.create_loss(args, prior=distPU_model.prior)

    # Warmup
    warm_up_lr = config.get('warm_up_lr', DEFAULT_CONFIG.get('warm_up_lr', None) if USE_DEFAULT_CONFIG else None)
    warm_up_weight_decay = config.get('warm_up_weight_decay', DEFAULT_CONFIG.get('warm_up_weight_decay', None) if USE_DEFAULT_CONFIG else None)
    warm_up_epochs = config.get('warm_up_epochs', DEFAULT_CONFIG.get('warm_up_epochs', None) if USE_DEFAULT_CONFIG else None)

    optimizer = torch.optim.Adam(
        distPU_model.parameters(), lr=warm_up_lr, weight_decay=warm_up_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, warm_up_epochs)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}
    for epoch in range(warm_up_epochs):
        scores_dict = {split: {} for split in dataloaders_dict.keys()}
        scores_dict_item = distPU_model.train_one_epoch(epoch=epoch, dataloader=dataloaders_dict['train'], loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
        scores_dict['train'].update(scores_dict_item)
        all_scores_dict['train']['epochs'].append(epoch)

        for split in ['val', 'test']:
            scores_dict_item = distPU_model.validate(dataloaders_dict[split], loss_fn=loss_fn, model=distPU_model.model)
            scores_dict[split].update(scores_dict_item)
            all_scores_dict[split]['epochs'].append(epoch)

        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Warmup Epoch {epoch}: {scores_dict}")

    # Training
    mixup_dataset = lpu.models.distPU.MixupDataset()
    mixup_dataset.update_psudos(dataloaders_dict['train'], distPU_model.model, distPU_model.device)

    lr = config.get('lr', DEFAULT_CONFIG.get('lr', None) if USE_DEFAULT_CONFIG else None)
    weight_decay = config.get('weight_decay', DEFAULT_CONFIG.get('weight_decay', None) if USE_DEFAULT_CONFIG else None)
    pu_epochs = config.get('pu_epochs', DEFAULT_CONFIG.get('pu_epochs', None) if USE_DEFAULT_CONFIG else None)

    optimizer = torch.optim.Adam(
        distPU_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pu_epochs, 0.7 * lr)

    co_entropy = 0
    for epoch in range(pu_epochs):
        co_entropy = update_co_entropy(config, epoch)
        LOG.info(f"Updating co-entropy: {co_entropy:.5f}")

        LOG.info("Training with mixup")
        scores_dict_item = distPU_model.train_mixup_one_epoch(epoch=epoch, dataloader=dataloaders_dict['train'], loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, mixup_dataset=mixup_dataset, co_entropy=co_entropy)
        scores_dict['train'].update(scores_dict_item)
        all_scores_dict['train']['epochs'].append(epoch)

        for split in ['val', 'test']:
            scores_dict_item = distPU_model.validate(dataloaders_dict[split], loss_fn=loss_fn, model=distPU_model.model)
            scores_dict[split].update(scores_dict_item)
            all_scores_dict[split]['epochs'].append(epoch)

        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Training Epoch {epoch}: {scores_dict}")

    lpu.utils.plot_utils.plot_scores(all_scores_dict, loss_type='overall_loss')

if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    main()