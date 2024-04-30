
import copy
import math
import sys
import types

sys.path.append('lpu/external_libs/distPU')
sys.path.append('lpu/external_libs/distPU/dataTools')

from matplotlib import pyplot as plt
import torch.nn
import torch.utils.data
import numpy as np

import lpu.constants
import lpu.datasets.dataset_utils
import lpu.datasets.LPUDataset
import lpu.external_libs
import lpu.external_libs.distPU
import lpu.external_libs.distPU.dataTools
import lpu.external_libs.distPU.dataTools.factory
import lpu.external_libs.distPU.losses
import lpu.external_libs.distPU.losses.factory
import lpu.external_libs.distPU.train
import lpu.models.distPU
import lpu.utils.auxiliary_models
import lpu.utils.utils_general



    
LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
BATCH_SIZE = 32
HOLDOUT_RATIO = 0.05
VAL_RATIO = 0.10
TEST_RATIO = 0.35


def update_co_entropy(config, epoch):
    co_entropy = (1-math.cos((float(epoch)/config['pu_epochs']) * (math.pi/2))) * config['co_entropy']
    return co_entropy


def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/distPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    args = types.SimpleNamespace(**config)
    


    ###########################################################################
    # START: data preparation
    ###########################################################################
    dataloaders_dict = lpu.datasets.dataset_utils.create_dataloaders_dict(config, transform=None, target_transform=None)


    ###########################################################################
    # START: warmup
    ###########################################################################
    if config['dataset_kind'] == 'LPU':
        dim = dataloaders_dict['train'].dataset.X.shape[1]
    elif config['dataset_kind'] == 'distPU':
        dim = torch.flatten(dataloaders_dict['train'].dataset.X, 1).shape[1]

    distPU_model = lpu.models.distPU.distPU(config, dim)
    distPU_model.set_C(copy.deepcopy(dataloaders_dict['train']))
    # config['alpha'] = dataloaders_dict['train'].dataset.y.mean().item()

    loss_fn = lpu.models.distPU.create_loss(args, prior=distPU_model.prior)

    # obtain optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            distPU_model.parameters(), lr=args.warm_up_lr,
            weight_decay=args.warm_up_weight_decay
        )
    else:
        raise NotImplementedError("The optimizer: {} is not defined!".format(args.optimizer))

    if args.schedular == 'cos-ann':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pu_epochs)
    else:
        raise NotImplementedError("The schedular: {} is not defined!".format(args.schedular))

    for epoch in range(args.warm_up_epochs):
        distPU_model.train_one_epoch(epoch=epoch, dataloader=dataloaders_dict['train'], loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
        print(distPU_model.validate(dataloaders_dict['val'], loss_fn=loss_fn, output_model=distPU_model.model))

    ###########################################################################
    # START: training
    ###########################################################################


    mixup_dataset = lpu.models.distPU.MixupDataset()
    mixup_dataset.update_psudos(dataloaders_dict['train'], distPU_model.model, distPU_model.device)

    optimizer = torch.optim.Adam(
            distPU_model.parameters(), lr=config['lr'],
            weight_decay=args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.pu_epochs, 0.7*args.lr)

    co_entropy = 0
    best_acc = 0

    for epoch in range(args.pu_epochs):
        co_entropy = update_co_entropy(config, epoch)
        print('==> updating co-entropy: {:.5f}'.format(co_entropy))

        print('==> training with mixup')
        distPU_model.train_mixup_one_epoch(epoch=epoch, dataloader=dataloaders_dict['train'], loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, mixup_dataset=mixup_dataset, co_entropy=co_entropy)
        print (distPU_model.validate(dataloaders_dict['val'], loss_fn=loss_fn, output_model=distPU_model.model))
        





if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    main()
