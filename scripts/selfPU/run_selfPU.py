import copy
import logging
import os
import sys
import types 

import lpu.external_libs.Self_PU.datasets
import lpu.external_libs.Self_PU.models

sys.path.append('lpu/external_libs/self_PU')
import numpy as np
import torch.backends.cudnn
import torch.utils.data
import torch
import unittest.mock

import lpu.external_libs
import lpu.external_libs.Self_PU
import lpu.external_libs.Self_PU.train
import lpu.external_libs.Self_PU.utils
import lpu.models.selfPU


LOG = logging.getLogger(__name__)

# import torch.optim

import lpu.constants
import lpu.datasets.LPUDataset
import lpu.datasets.dataset_utils
import lpu.utils.utils_general
import lpu.external_libs.Self_PU.functions

LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
TRAIN_VAL_RATIO = .1
HOLD_OUT_SIZE = None
TRAIN_TEST_RATIO = .5

def create_model(ema=False, dim=4096):
    model = lpu.external_libs.Self_PU.models.MultiLayerPerceptron(dim)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

global single_epoch_steps
def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/selfPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False, invert_l=False)
    train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=len(lpu_dataset), hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    # nnPU_model = lpu.models.nnPU.nnPU(config)
    args = types.SimpleNamespace(**config)

    num_workers = config.get('num_workers', 4)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 200)
    learning_rate = config.get('learning_rate', 5e-4)
    replacement = config.get('replacement', True)
    self_paced_type = config.get('self_paced_type', False)
    top = config.get('top', 0.5)
    seed = config.get('seed', None)
    increasing = config.get('increasing', True)
    device = config.get('device', 'cpu')
    global single_epoch_steps
    single_epoch_steps = 0

    selfPU_model = lpu.models.selfPU.selfPU(config)
    selfPU_model.set_C(train_loader)
    # with (unittest.mock.patch('lpu.external_libs.Self_PU.train.args', args),  unittest.mock.patch.object(torch.Tensor, 'cuda', lambda x: x.to(device)),
            # unittest.mock.patch.object(torch.Tensor, 'float', lambda x: x.to(lpu.constants.DTYPE))): 
    val_loader_copy = copy.deepcopy(val_loader) 
    with unittest.mock.patch('lpu.external_libs.Self_PU.train.args', args):
        criterion = lpu.external_libs.Self_PU.train.get_criterion()
    consistency_criterion = lpu.external_libs.Self_PU.train.losses.softmax_mse_loss
    if device in ['cuda', 'gpu']:
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
    trainX, trainL, trainY = next(iter(train_loader))
    testX, testL, testY = next(iter(test_loader))
    valX, valL, valY = next(iter(val_loader))

    with unittest.mock.patch.object(np, 'float32', lpu.constants.NUMPY_DTYPE):
        dataset_train_clean = lpu.models.selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='train',ids=[],
            increasing=increasing, replacement=replacement, mode=self_paced_type, top = top, type="clean", seed = seed)

        dataset_train_noisy = lpu.models.selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='train', 
            increasing=increasing, replacement=replacement, mode=self_paced_type, top = top, type="noisy", seed = seed)

        dataset_train_noisy.copy(dataset_train_clean) # 和clean dataset使用相同的随机顺序
        dataset_train_noisy.reset_ids() # 让初始化的noisy dataset使用全部数据

    dataset_train_noisy.copy(dataset_train_clean) # 和clean dataset使用相同的随机顺序
    dataset_train_noisy.reset_ids() # 让初始化的noisy dataset使用全部数据

    with unittest.mock.patch.object(np, 'float32', lpu.constants.NUMPY_DTYPE):
        dataset_test = lpu.models.selfPU.selfPUModifiedDataset(trainX, trainY, trainL, testX, testY, testL, split='test', 
            increasing=increasing, replacement=replacement, mode=self_paced_type, top = top, type="clean", seed = seed)

        dataset_val = lpu.models.selfPU.selfPUModifiedDataset(trainX, trainY, trainL, valX, valY, valL, split='test',
            increasing=increasing, replacement=replacement, mode=self_paced_type, top = top, type="clean", seed = seed)

    if len(dataset_train_clean) > 0:
        dataloader_train_clean = torch.utils.data.DataLoader(dataset_train_clean, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    else:
        dataloader_train_clean = None
    
    if len(dataset_train_noisy) > 0:
        dataloader_train_noisy = torch.utils.data.DataLoader(dataset_train_noisy, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    else:
        dataloader_train_noisy = None
    
    if len(dataset_val):
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)
    else:
        dataloader_val = None

    assert np.all(dataset_train_noisy.X == dataset_train_clean.X)
    assert np.all(dataset_train_noisy.Y == dataset_train_clean.Y)
    assert np.all(dataset_train_noisy.oids == dataset_train_clean.oids)
    assert np.all(dataset_train_noisy.T == dataset_train_clean.T)

    dim = torch.flatten(torch.tensor(dataset_train_clean.X), start_dim=1).shape[-1]
    selfPU_model.model = create_model(dim=dim)
    selfPU_model.ema_model = create_model(ema = True, dim=dim)
    params_list = [{'params': selfPU_model.model.parameters()}] 
    optimizer = torch.optim.Adam(params_list, lr=args.lr,
       weight_decay=args.weight_decay
    ) 
    model_dir = config.get('model_dir', 'lpu/scripts/selfPU')
    stats_ = lpu.external_libs.Self_PU.functions.stats(model_dir, 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min = learning_rate * 0.2)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.6)
    best_acc = 0

    single_epoch_steps = len(dataloader_train_noisy) + 1
    if device in ['cuda', 'gpu']:
        selfPU_model.model = selfPU_model.model.cuda()
        selfPU_model.ema_model = selfPU_model.ema_model.cuda()


    val = []
    for epoch in range(num_epochs):
        # with (unittest.mock.patch('lpu.external_libs.Self_PU.train.args', args), unittest.mock.patch.object(torch.Tensor, 'cuda', lambda x: x.to(device)),
        #       unittest.mock.patch.object(torch.Tensor, 'float', lambda x: x.to(lpu.constants.DTYPE))): 
        with unittest.mock.patch('lpu.external_libs.Self_PU.train.args', args):
            print("Self paced status: {}".format(lpu.external_libs.Self_PU.train.check_self_paced(epoch)))
            print("Mean teacher status: {}".format(lpu.external_libs.Self_PU.train.check_mean_teacher(epoch)))
            print("Noisy status: {}".format(lpu.external_libs.Self_PU.train.check_noisy(epoch)))

            if lpu.external_libs.Self_PU.train.check_mean_teacher(epoch) and (not lpu.external_libs.Self_PU.train.check_mean_teacher(epoch - 1)) and not switched:
                selfPU_model.model.eval()
                selfPU_model.ema_model.eval()
                selfPU_model.ema_model.load_state_dict(selfPU_model.model.state_dict())
                switched = True
                print("SWITCHED!")
                lpu.external_libs.Self_PU.train.validate(dataloader_val, selfPU_model.model, selfPU_model.ema_model, criterion, consistency_criterion, epoch)
                lpu.external_libs.Self_PU.train.validate(dataloader_val, selfPU_model.ema_model, selfPU_model.model, criterion, consistency_criterion, epoch)
                selfPU_model.model.train()
                selfPU_model.ema_model.train()
            if epoch == 0:
                switched = False

            if (not lpu.external_libs.Self_PU.train.check_mean_teacher(epoch)) and lpu.external_libs.Self_PU.train.check_mean_teacher(epoch - 1) and not switched:
                selfPU_model.model.eval()
                selfPU_model.ema_model.eval()
                selfPU_model.model.load_state_dict(selfPU_model.ema_model.state_dict())
                switched = True
                print("SWITCHED!")
                lpu.external_libs.Self_PU.train.validate(dataloader_val, selfPU_model.model, selfPU_model.ema_model, criterion, consistency_criterion, epoch)
                lpu.external_libs.Self_PU.train.validate(dataloader_val, selfPU_model.ema_model, selfPU_model.model, criterion, consistency_criterion, epoch)
                selfPU_model.model.train()
                selfPU_model.ema_model.train()
            trainPacc, trainNacc, trainPNacc = lpu.external_libs.Self_PU.train.train(dataloader_train_clean, dataloader_train_noisy, selfPU_model.model, selfPU_model.ema_model, criterion, consistency_criterion, optimizer, scheduler, epoch, self_paced_pick=len(dataset_train_clean))
            valPacc, valNacc, valPNacc = lpu.external_libs.Self_PU.train.validate(dataloader_val, selfPU_model.model, selfPU_model.ema_model, criterion, consistency_criterion, epoch)
            
        val.append(valPNacc)
        #validate_2(dataloader_test, selfPU_model.model, selfPU_model.ema_model, criterion, consistency_criterion, epoch)
        stats_._update(trainPacc, trainNacc, trainPNacc, valPacc, valNacc, valPNacc)

        is_best = valPNacc > best_acc
        best_acc = max(valPNacc, best_acc)
        filename = []
        filename.append(os.path.join(model_dir, 'checkpoint.pth.tar'))
        filename.append(os.path.join(model_dir, 'model_best.pth.tar'))
        with unittest.mock.patch('lpu.external_libs.Self_PU.train.args', args):
            if (lpu.external_libs.Self_PU.train.check_self_paced(epoch)) and (epoch - args.self_paced_start) % args.self_paced_frequency == 0:

                dataloader_train_clean, dataloader_train_noisy = lpu.external_libs.Self_PU.train.update_dataset(selfPU_model.model, selfPU_model.ema_model, dataset_train_clean, dataset_train_noisy, epoch)

        lpu.external_libs.Self_PU.functions.plot_curve(stats_, model_dir, 'selfPU_model.model', True)
        lpu.external_libs.Self_PU.train.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': selfPU_model.model.state_dict(),
            'best_prec1': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename)
        dataset_train_noisy.shuffle()

        #dataloader_train_clean = DataLoader(dataset_train_clean, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        dataloader_train_noisy = torch.utils.data.DataLoader(dataset_train_noisy, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)
        print(selfPU_model.validate(val_loader_copy))
    print(best_acc)
    print(val)

if __name__ == "__main__":
    main()

