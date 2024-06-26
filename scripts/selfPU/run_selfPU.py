import copy
import random

import sys
sys.path.append('LPU/external_libs/PU_learning')
sys.path.append('LPU/external_libs/PU_learning/data_helper')
sys.path.append('LPU/models/selfPU/selfPU')

import torchvision
import torch.backends.cudnn


import LPU.external_libs.Self_PU
import LPU.external_libs.Self_PU.cifar_datasets
import LPU.external_libs.Self_PU.datasets
import LPU.external_libs.Self_PU.functions
import LPU.external_libs.Self_PU.mean_teacher
import LPU.external_libs.Self_PU.mean_teacher.losses
import LPU.external_libs.Self_PU.meta_models
import LPU.external_libs.Self_PU.models
import LPU.external_libs.Self_PU.utils
import LPU.external_libs.Self_PU.utils.util
import LPU.models.selfPU.selfPU
import LPU.models.selfPU.dataset_utils




from matplotlib import pyplot as plt
import torch.nn
import torch.utils.data
import numpy as np

import LPU.constants
import LPU.utils.dataset_utils
import LPU.datasets.LPUDataset
import LPU.models.MPE.MPE
import LPU.utils.plot_utils
import LPU.utils.utils_general

import sklearn.model_selection

import tempfile
import os

LOG = LPU.utils.utils_general.configure_logger(__name__)

# Optional dynamic import for Ray
try:
    import ray.util.client
    import ray.train
    RAY_AVAILABLE = True
except ImportError:
    LOG.warning("Ray is not available. Please install Ray to enable distributed training.")
    RAY_AVAILABLE = False





def get_criterion(config):
    weights = [float(config['weight']), 1.0]
    class_weights = torch.FloatTensor(weights)

    class_weights = class_weights.to(config['device'])
    if config['loss'] == 'Xent':
        criterion = LPU.external_libs.Self_PU.utils.util.PULoss(Probability_P=0.49, loss_fn="Xent")
    elif config['loss'] == 'nnPU':
        criterion = LPU.external_libs.Self_PU.utils.util.PULoss(Probability_P=0.49)
    elif config['loss'] == 'Focal':
        class_weights = torch.FloatTensor(weights).to(config['device'])
        criterion = LPU.external_libs.Self_PU.utils.util.FocalLoss(gamma=0, weight=class_weights, one_hot=False)
    elif config['loss'] == 'uPU':
        criterion = LPU.external_libs.Self_PU.utils.util.PULoss(Probability_P=0.49, nnPU=False)
    elif config['loss'] == 'Xent_weighted':
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    return criterion

def identity(x):
    return x

def make_transformations(dataset_name):
    transformations = {
        'cifar': {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        },
        'mnist': {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                torch.flatten
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                torch.flatten
            ]),
        },
        'animal_no_animal':
        {
            'train': identity,
            'val': identity
        }
        
    }

    return transformations[dataset_name]

def train_model(config=None, dataloaders_dict=None, with_ray=False):
    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(LPU.models.selfPU.selfPU.DEFAULT_CONFIG, config)


    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
        # setting the seed for the training
        LPU.utils.utils_general.set_seed(random_state)



    if dataloaders_dict is None:
        if config['dataset_kind'] == 'LPU':
            dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config, target_transform=LPU.utils.dataset_utils.one_zero_to_minus_one_one,
                                                                       label_transform=LPU.utils.dataset_utils.one_zero_to_minus_one_one)
        else:
            dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    

    

    # torch.manual_seed(config['seed'])
    # if config['seed'] is not None:
    #     random.seed(config['seed'])
    #     torch.manual_seed(config['seed'])
    #     torch.backends.cudnn.deterministic = True
        
    criterion = get_criterion(config)
    criterion_meta = LPU.external_libs.Self_PU.utils.util.PULoss(Probability_P=0.49, loss_fn="sigmoid_eps")

    data_transforms = make_transformations(config['dataset_name'])
    if config['dataset_kind'] == 'SelfPU':
        (trainX, trainY), (testX, testY) = LPU.models.selfPU.dataset_utils.get_dataset(config['dataset_name'], path=config['datapath'])
        # max_size = 5000
        max_size = -1
        trainX = trainX[:max_size]
        trainY = trainY[:max_size]
        testX = testX[:max_size]
        testY = testY[:max_size]

        train_val_ratio = config['ratios']['train'] / (config['ratios']['train'] + config['ratios']['val'])

        trainX, valX, trainY, valY = sklearn.model_selection.train_test_split(trainX, trainY, test_size=train_val_ratio)
        trainY, valY, testY = LPU.models.selfPU.dataset_utils.binarize_class(trainY, valY, testY, dataset_name=config['dataset_name'])

        n_u_train = len(trainY)# - config['num_p']
        n_u_val = len(valY)# - config['num_p']
        n_u_test = len(testY)# - config['num_p']
        
        X_train, Y_train, T_train, oids_train, prior_train = LPU.models.selfPU.dataset_utils.make_dataset(trainX=trainX, trainY=trainY, n_labeled=config['num_p'], n_unlabeled=n_u_train)
        X_val, Y_val, T_val, oids_val, prior_val = LPU.models.selfPU.dataset_utils.make_dataset(trainX=valX, trainY=valY, n_labeled=config['num_p'], n_unlabeled=n_u_val)
        X_test, Y_test, T_test, oids_test, prior_test = LPU.models.selfPU.dataset_utils.make_dataset(trainX=testX, trainY=testY, n_labeled=config['num_p'], n_unlabeled=n_u_test)
    elif config['dataset_kind'] == 'LPU':        
        X_train = dataloaders_dict['train'].dataset.X.detach().cpu().numpy()
        T_train = dataloaders_dict['train'].dataset.y.detach().cpu().numpy()
        Y_train = dataloaders_dict['train'].dataset.l.detach().cpu().numpy()
        oids_train = np.arange(len(Y_train))
        prior_train = Y_train.mean()

        X_val = dataloaders_dict['val'].dataset.X.detach().cpu().numpy()
        T_val = dataloaders_dict['val'].dataset.y.detach().cpu().numpy()
        Y_val = dataloaders_dict['val'].dataset.l.detach().cpu().numpy()
        oids_val = np.arange(len(Y_val))
        prior_val = Y_val.mean()

        X_test = dataloaders_dict['test'].dataset.X.detach().cpu().numpy()
        T_test = dataloaders_dict['test'].dataset.y.detach().cpu().numpy()
        Y_test = dataloaders_dict['test'].dataset.l.detach().cpu().numpy()
        oids_test = np.arange(len(Y_test))
        prior_test = Y_test.mean()

    

    dataset_train1_clean = LPU.models.selfPU.dataset_utils.SelfPUDataset(
        X=X_train, Y=Y_train, T=T_train, oids=oids_train, prior=prior_train, ids=[], increasing=config['increasing'], replacement=config['replacement'], mode=config['self_paced_type'], top = config['top1'], transform = data_transforms['train'], type="clean")

    dataset_train1_noisy = LPU.models.selfPU.dataset_utils.SelfPUDataset(
        X=X_train, Y=Y_train, T=T_train, oids=oids_train, prior=prior_train,
        increasing=config['increasing'], replacement=config['replacement'], mode=config['self_paced_type'], top = config['top1'], transform = data_transforms['train'], type="noisy")

    dataset_train1_noisy.copy(dataset_train1_clean) # 和clean dataset使用相同的随机顺序
    dataset_train1_noisy.reset_ids() # 让初始化的noisy dataset使用全部数据

    dataset_test = LPU.models.selfPU.dataset_utils.SelfPUDataset(
        X=X_test, Y=Y_test, T=T_test, oids=oids_test, prior=prior_test,
        increasing=config['increasing'], replacement=config['replacement'], mode=config['self_paced_type'], transform = data_transforms['val'], type="clean")

    dataset_val = LPU.models.selfPU.dataset_utils.SelfPUDataset(
        X=X_val, Y=Y_val, T=T_val, oids=oids_val, prior=prior_val,
        increasing=config['increasing'], replacement=config['replacement'], mode=config['self_paced_type'], transform = data_transforms['val'], type="clean")
    dataset_train2_noisy = LPU.models.selfPU.dataset_utils.SelfPUDataset(
        X=X_train, Y=Y_train, T=T_train, oids=oids_train, prior=prior_train,
        increasing=config['increasing'], replacement=config['replacement'], mode=config['self_paced_type'], 
        transform = data_transforms['train'], top = config['top2'], type="noisy")
    dataset_train2_clean = LPU.models.selfPU.dataset_utils.SelfPUDataset(
        X=X_train, Y=Y_train, T=T_train, oids=oids_train, prior=prior_train,
        increasing=config['increasing'], replacement=config['replacement'], mode=config['self_paced_type'], 
        transform = data_transforms['train'], top = config['top2'], type="clean", ids=[])
    
    dataset_train2_noisy.copy(dataset_train1_noisy)
    dataset_train2_noisy.reset_ids()
    dataset_train2_clean.copy(dataset_train1_clean)

    assert np.all(dataset_train1_clean.X == dataset_train1_noisy.X)
    assert np.all(dataset_train2_clean.X == dataset_train1_noisy.X)
    assert np.all(dataset_train2_noisy.X == dataset_train1_noisy.X)

    assert np.all(dataset_train1_clean.Y == dataset_train1_noisy.Y)
    assert np.all(dataset_train2_clean.Y == dataset_train1_noisy.Y)
    assert np.all(dataset_train2_noisy.Y == dataset_train1_noisy.Y)

    assert np.all(dataset_train1_clean.T == dataset_train1_noisy.T)
    assert np.all(dataset_train2_clean.T == dataset_train1_noisy.T)
    assert np.all(dataset_train2_noisy.T == dataset_train1_noisy.T)

    criterion.update_p(0.4)
    dataloader_train1_clean = None
    dataloader_train1_noisy = torch.utils.data.DataLoader(dataset_train1_noisy, batch_size=config['batch_size']['train'], num_workers=config['workers'], shuffle=False, pin_memory=True)
    dataloader_train2_clean = None
    dataloader_train2_noisy = torch.utils.data.DataLoader(dataset_train2_noisy, batch_size=config['batch_size']['train'], num_workers=config['workers'], shuffle=False, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config['batch_size']['val'], shuffle=False, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size']['test'], shuffle=False, pin_memory=True)
    consistency_criterion = LPU.external_libs.Self_PU.mean_teacher.losses.softmax_mse_loss
    selfPU_model = LPU.models.selfPU.selfPU.selfPU(config=config)

    selfPU_model.model1 = selfPU_model.create_model().to(config['device'])
    selfPU_model.model2 = selfPU_model.create_model().to(config['device'])
    selfPU_model.ema_model1 = selfPU_model.create_model(ema = True).to(config['device'])
    selfPU_model.ema_model2 = selfPU_model.create_model(ema = True).to(config['device'])
    for name, param in selfPU_model.model1.named_parameters():
        print ("name: ", name)
        if not param.is_leaf:
            print(f"Parameter '{name}' is not a leaf tensor.")    
    for param in selfPU_model.model1.parameters():
        if not param.is_leaf:
            print("Found a non-leaf parameter:")
            # print(param)
            print(f"Gradient function:) {param.grad_fn}")    
    optimizer1 = torch.optim.Adam(selfPU_model.model1.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    optimizer2 = torch.optim.Adam(selfPU_model.model2.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    stats_ = LPU.external_libs.Self_PU.functions.stats(config['modeldir'], 0)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, config['epochs'])
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, config['epochs'])

    best_acc1 = 0
    best_acc2 = 0
    best_acc3 = 0
    best_acc4 = 0
    best_acc = 0
    best_val_loss = float('inf')
    best_epoch = -1
    scores_dict = {}
    best_model_state = selfPU_model.state_dict()
    best_scores_dict = {}
    all_scores_dict = {split: {'epochs': []} for split in ['train', 'val']}

    for epoch in range(config['epochs']):
        print("Self paced status: {}".format(selfPU_model.check_self_paced(epoch)))
        print("Mean Teacher status: {}".format(selfPU_model.check_mean_teacher(epoch)))
        if selfPU_model.check_mean_teacher(epoch) and not selfPU_model.check_mean_teacher(epoch - 1) and not selfPU_model.switched:
            selfPU_model.ema_model1.load_state_dict(selfPU_model.model1.state_dict())
            selfPU_model.ema_model2.load_state_dict(selfPU_model.model2.state_dict())
            selfPU_model.switched = True
            LOG.info("Switched to Mean Teacher")
        if not selfPU_model.check_self_paced(epoch):
            scores_dict['train'] = selfPU_model.train(dataloader_train1_clean, dataloader_train1_noisy, dataloader_train2_clean, dataloader_train2_noisy, selfPU_model.model1, selfPU_model.model2, selfPU_model.ema_model1, selfPU_model.ema_model2, criterion, consistency_criterion, optimizer1, scheduler1, optimizer2, scheduler2, epoch)
            scores_dict['train']['self_paced'] = True
        else:
            scores_dict['train'] = selfPU_model.train_with_meta(dataloader_train1_clean, dataloader_train1_noisy, dataloader_train2_clean, dataloader_train2_noisy, dataloader_val,selfPU_model.model1, selfPU_model.model2, selfPU_model.ema_model1, selfPU_model.ema_model2, criterion_meta, consistency_criterion, optimizer1, scheduler1, optimizer2, scheduler2, epoch)
            scores_dict['train']['self_paced'] = False
        all_scores_dict['train']['epochs'].append(epoch)
            
        scores_dict['val'] = selfPU_model.validate(dataloader_val, loss_fn=criterion, model=selfPU_model.model1)
        all_scores_dict['val']['epochs'].append(epoch)
        for split in ['train', 'val']:
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)
        LOG.info(f"Epoch {epoch}: {scores_dict}")
        
        # models = [model1, model2, ema_model1, ema_model2]

        if (selfPU_model.check_self_paced(epoch)) and (epoch - config['self_paced_start']) % config['self_paced_frequency'] == 0:

            dataloader_train1_clean, dataloader_train1_noisy, dataloader_train2_clean, dataloader_train2_noisy = selfPU_model.update_dataset(selfPU_model.model1, selfPU_model.model2, selfPU_model.ema_model1, selfPU_model.ema_model2, dataset_train1_clean, dataset_train1_noisy, dataset_train2_clean, dataset_train2_noisy, epoch, )

        dataset_train1_noisy.shuffle()
        dataset_train2_noisy.shuffle()
        dataloader_train1_noisy = torch.utils.data.DataLoader(dataset_train1_noisy, batch_size=config['batch_size']['train'], num_workers=config['workers'], shuffle=False, pin_memory=True)
        dataloader_train2_noisy = torch.utils.data.DataLoader(dataset_train2_noisy, batch_size=config['batch_size']['train'], num_workers=config['workers'], shuffle=False, pin_memory=True)

        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            best_scores_dict = copy.deepcopy(scores_dict)
            best_model_state = copy.deepcopy(selfPU_model.state_dict())

        # Add checkpointing code
        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch,
                     "model_state": selfPU_model.state_dict(),
                     "config": config,
                     "model1": selfPU_model.model1.state_dict(),
                     "model2": selfPU_model.model2.state_dict(),
                     "ema_model1": selfPU_model.ema_model1.state_dict(),
                     "ema_model2": selfPU_model.ema_model2.state_dict(),
                     "dataloader_test": dataloader_test,},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                ray.train.report(metrics={
                    'val_overall_loss': scores_dict['val']['overall_loss'],
                    'val_y_auc': scores_dict['val']['y_auc'],
                    'val_y_accuracy': scores_dict['val']['y_accuracy'],
                    'val_y_APS': scores_dict['val']['y_APS'],
                    'epoch': epoch,}, checkpoint=ray.train.Checkpoint.from_directory(tempdir))

    LOG.info(f"Best epoch: {best_epoch}, Best validation overall_loss: {best_val_loss:.5f}")

    model = selfPU_model
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
        best_scores_dict['test'] = model.validate(dataloader_test, loss_fn=criterion, model=model.model1)
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
    LPU.utils.plot_utils.plot_scores(results, best_epoch=best_epoch)