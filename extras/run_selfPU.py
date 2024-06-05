import copy
import logging
import os
import sys

import lpu.utils.auxiliary_models
sys.path.append('lpu/external_libs/Self_PU')
sys.path.append('lpu/external_libs/Self_PU/mean_teacher')
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import unittest.mock

import types

import lpu.external_libs.Self_PU.datasets
import lpu.external_libs.Self_PU.models
import lpu.external_libs.Self_PU.train
import lpu.external_libs.Self_PU.utils
import lpu.models.old_selfPU
import lpu.utils.plot_utils
import lpu.utils.utils_general

import lpu.constants
import lpu.datasets.LPUDataset
import lpu.utils.dataset_utils

LOG = lpu.utils.utils_general.configure_logger(__name__)

def create_model(ema=False, dim=4096):
    model = lpu.utils.auxiliary_models.MultiLayerPerceptron(input_dim=dim, output_dim=1)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

    return criterion

def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/selfPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    dataloaders_dict = lpu.utils.dataset_utils.create_dataloaders_dict(config)
    args = types.SimpleNamespace(**config)

    device = config.get('device', 'cpu')
    # val_loader_copy = copy.deepcopy(dataloaders_dict['val'])
    with unittest.mock.patch('lpu.external_libs.Self_PU.train.args', args):
        criterion = lpu.external_libs.Self_PU.train.get_criterion()
    consistency_criterion = lpu.external_libs.Self_PU.train.losses.softmax_mse_loss
    if device in ['cuda', 'gpu']:
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

    data_dict = {split: {'X': [], 'L': [], 'Y': []} for split in dataloaders_dict.keys()}
    for split in dataloaders_dict.keys():
        for i, (X, L, Y, _) in enumerate(dataloaders_dict[split]):
            data_dict[split]['X'].append(X)
            data_dict[split]['L'].append(L)
            data_dict[split]['Y'].append(Y)
    for split in dataloaders_dict.keys():
        data_dict[split]['X'] = torch.cat(data_dict[split]['X'], axis=0)
        data_dict[split]['L'] = torch.cat(data_dict[split]['L'], axis=0)
        data_dict[split]['Y'] = torch.cat(data_dict[split]['Y'], axis=0)

            
    trainX, trainL, trainY = data_dict['train']['X'], data_dict['train']['L'], data_dict['train']['Y']
    testX, testL, testY = data_dict['test']['X'], data_dict['test']['L'], data_dict['test']['Y']
    valX, valL, valY = data_dict['val']['X'], data_dict['val']['L'], data_dict['val']['Y']

    # first dataset
    dataset_train1_clean = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='train', ids=[],
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top1, type="clean", seed=args.seed)

    dataset_train1_noisy = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='train',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top1, type="noisy", seed=args.seed)

    dataset_train1_noisy.copy(dataset_train1_clean)
    dataset_train1_noisy.reset_ids()

    dataset_test = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, testX, testY, testL, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top, type="clean", seed=args.seed)

    dataset_val = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top, type="clean", seed=args.seed)

    if len(dataset_train1_clean) > 0:
        dataloader_train1_clean = torch.utils.data.DataLoader(dataset_train1_clean, batch_size=args.batch_size['train'], num_workers=args.num_workers, shuffle=True, pin_memory=True)
    else:
        dataloader_train1_clean = None

    if len(dataset_train1_noisy) > 0:
        dataloader_train1_noisy = torch.utils.data.DataLoader(dataset_train1_noisy, batch_size=args.batch_size['train'], num_workers=args.num_workers, shuffle=False, pin_memory=True)
    else:
        dataloader_train1_noisy = None

    assert np.all(dataset_train1_clean.X == dataset_train1_noisy.X)
    assert np.all(dataset_train2_clean.X == dataset_train1_noisy.X)
    assert np.all(dataset_train2_noisy.X == dataset_train1_noisy.X)

    # second dataset
    dataset_train2_clean = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='train', ids=[],
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top2, type="clean", seed=args.seed)

    dataset_train2_noisy = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='train',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top2, type="noisy", seed=args.seed)

    dataset_train2_noisy.copy(dataset_train1_clean)
    dataset_train2_noisy.reset_ids()

    dataset_test = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, testX, testY, testL, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top, type="clean", seed=args.seed)

    dataset_val = lpu.models.old_selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='test',
        increasing=args.increasing, replacement=args.replacement, mode=args.self_paced_type, top=args.top, type="clean", seed=args.seed)

    if len(dataset_train2_clean) > 0:
        dataloader_train2_clean = torch.utils.data.DataLoader(dataset_train2_clean, batch_size=args.batch_size['train'], num_workers=args.num_workers, shuffle=True, pin_memory=True)
    else:
        dataloader_train2_clean = None

    if len(dataset_train2_noisy) > 0:
        dataloader_train2_noisy = torch.utils.data.DataLoader(dataset_train2_noisy, batch_size=args.batch_size['train'], num_workers=args.num_workers, shuffle=False, pin_memory=True)
    else:
        dataloader_train2_noisy = None



    if len(dataset_val):
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size['val'], num_workers=0, shuffle=False, pin_memory=True)
    else:
        dataloader_val = None

    if len(dataset_test):
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size['test'], num_workers=0, shuffle=False, pin_memory=True)
    else:
        dataloader_test = None

    assert np.all(dataset_train1_clean.X == dataset_train1_noisy.X)
    assert np.all(dataset_train2_clean.X == dataset_train1_noisy.X)
    assert np.all(dataset_train2_noisy.X == dataset_train1_noisy.X)

    assert np.all(dataset_train1_clean.Y == dataset_train1_noisy.Y)
    assert np.all(dataset_train2_clean.Y == dataset_train1_noisy.Y)
    assert np.all(dataset_train2_noisy.Y == dataset_train1_noisy.Y)

    assert np.all(dataset_train1_clean.T == dataset_train1_noisy.T)
    assert np.all(dataset_train2_clean.T == dataset_train1_noisy.T)
    assert np.all(dataset_train2_noisy.T == dataset_train1_noisy.T)

    selfPU_model = lpu.models.old_selfPU.selfPU(config, single_epoch_steps=len(dataloader_train1_noisy) + 1)
    selfPU_model.set_C(dataloaders_dict['holdout'])

    dim = torch.flatten(torch.tensor(dataset_train1_clean.X), start_dim=1).shape[-1]
    selfPU_model.model1 = create_model(dim=dim)
    selfPU_model.model2 = create_model(dim=dim)

    selfPU_model.ema_model1 = create_model(ema=True, dim=dim)
    selfPU_model.ema_model2 = create_model(ema=True, dim=dim)

    params_list1 = [{'params': selfPU_model.model1.parameters()}]
    params_list2 = [{'params': selfPU_model.model2.parameters()}]

    optimizer1 = torch.optim.Adam(params_list1, lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(params_list2, lr=args.lr, weight_decay=args.weight_decay)

    model_dir = config.get('model_dir', 'lpu/scripts/selfPU')
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, args.num_epochs, eta_min=args.lr * 0.2)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, args.num_epochs, eta_min=args.lr * 0.2)
    best_acc = 0

    if device in ['cuda', 'gpu']:
        selfPU_model.model1 = selfPU_model.model1.cuda()
        selfPU_model.ema_model1 = selfPU_model.ema_model.cuda()
        selfPU_model.model2 = selfPU_model.model2.cuda()
        selfPU_model.ema_model2 = selfPU_model.ema_model2.cuda()

    val = []
    test = []
    scores_dict = {}
    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}

    for epoch in range(args.num_epochs):
        print("Self paced status: {}".format(selfPU_model.check_self_paced(epoch)))
        print("Mean teacher status: {}".format(selfPU_model.check_mean_teacher(epoch)))
        print("Noisy status: {}".format(selfPU_model.check_noisy(epoch)))

        if selfPU_model.check_mean_teacher(epoch) and (not selfPU_model.check_mean_teacher(epoch - 1)) and not switched:
            selfPU_model.ema_model1.load_state_dict(selfPU_model.model1.state_dict())
            selfPU_model.ema_model2.load_state_dict(selfPU_model.model2.state_dict())

            switched = True
            print("SWITCHED!")
            selfPU_model.validate(dataloader_val, selfPU_model.model, selfPU_model.ema_model, criterion, consistency_criterion, epoch)
            selfPU_model.validate(dataloader_val, selfPU_model.ema_model, selfPU_model.model, criterion, consistency_criterion, epoch)
            selfPU_model.model.train()
            selfPU_model.ema_model.train()

        if epoch == 0:
            switched = False

        if (not selfPU_model.check_mean_teacher(epoch)) and selfPU_model.check_mean_teacher(epoch - 1) and not switched:
            selfPU_model.model.eval()
            selfPU_model.ema_model.eval()
            selfPU_model.model.load_state_dict(selfPU_model.ema_model.state_dict())
            switched = True
            print("SWITCHED!")
            selfPU_model.validate(val_loader=dataloader_val, model=selfPU_model.model, ema_model=selfPU_model.ema_model, criterion=criterion, consistency_criterion=consistency_criterion, epoch=epoch)
            selfPU_model.validate(val_loader=dataloader_val, model=selfPU_model.ema_model, ema_model=selfPU_model.model, criterion=criterion, consistency_criterion=consistency_criterion, epoch=epoch)
            selfPU_model.model.train()
            selfPU_model.ema_model.train()
        scores_dict['train'] = selfPU_model.train(clean_loader=dataloader_train1_clean, noisy_loader=dataloader_train1_noisy,model=selfPU_model.model, ema_model=selfPU_model.ema_model, criterion=criterion,  consistency_criterion=consistency_criterion, optimizer=optimizer, scheduler=scheduler, epoch=epoch, self_paced_pick=len(dataset_train1_clean))
        scores_dict['val'] = selfPU_model.validate(val_loader=dataloader_val, model=selfPU_model.model, ema_model=selfPU_model.ema_model, criterion=criterion, consistency_criterion=consistency_criterion, epoch=epoch)
        scores_dict['test'] = selfPU_model.validate(val_loader=dataloader_test, model=selfPU_model.model, ema_model=selfPU_model.ema_model, criterion=criterion, consistency_criterion=consistency_criterion, epoch=epoch)

        all_scores_dict['train']['epochs'].append(epoch)
        all_scores_dict['val']['epochs'].append(epoch)
        all_scores_dict['test']['epochs'].append(epoch)
        for split in dataloaders_dict.keys():
            if split == 'holdout':
                continue
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        # val.append(valPNacc)
        # test.append(testPNacc)
    
        # stats_._update(trainPacc, trainNacc, trainPNacc, valPacc, valNacc, valPNacc)

        # is_best = valPNacc > best_acc
        # best_acc = max(valPNacc, best_acc)
        # filename = []
        # filename.append(os.path.join(model_dir, 'checkpoint.pth.tar'))
        # filename.append(os.path.join(model_dir, 'model_best.pth.tar'))

        if (selfPU_model.check_self_paced(epoch)) and (epoch - config['self_paced_start']) % config['self_paced_frequency'] == 0:
            dataloader_train1_clean, dataloader_train1_noisy = selfPU_model.update_dataset(student=selfPU_model.model, teacher=selfPU_model.ema_model, dataset_train1_clean=dataset_train1_clean, dataset_train1_noisy=dataset_train1_noisy, epoch=epoch)


        dataloader_train1_noisy = torch.utils.data.DataLoader(dataset_train1_noisy, batch_size=config['batch_size']['train'], num_workers=config['num_workers'], shuffle=False, pin_memory=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config['batch_size']['train'], num_workers=0, shuffle=False, pin_memory=True)
        # scores_dict['val'].update(selfPU_model.validate(dataloader_val, selfPU_model.model, selfPU_model.ema_model, consistency_criterion=consistency_criterion))
        LOG.info(f"Epoch {epoch}: {scores_dict}")

    print(best_acc)
    print(val)

    lpu.utils.plot_utils.plot_scores(all_scores_dict, loss_type='overall_loss')

if __name__ == "__main__":
    # import warnings
    # warnings.simplefilter("error", category=UserWarning)
    main()