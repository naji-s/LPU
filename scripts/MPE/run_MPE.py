import copy
import tempfile
import os

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
import LPU.models.MPE.MPE
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



def train_model(config=None, dataloaders_dict=None, with_ray=False):

    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(LPU.models.MPE.MPE.DEFAULT_CONFIG, config)

    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
        # setting the seed for the training
        LPU.utils.utils_general.set_seed(random_state)



    if dataloaders_dict is None:
        dataloaders_dict = LPU.models.MPE.MPE.create_dataloaders_dict_mpe(config)
    


    criterion = torch.nn.CrossEntropyLoss()

    mpe_model = LPU.models.MPE.MPE.MPE(config)

    mpe_dataloaders_dict = LPU.models.MPE.MPE.create_dataloaders_dict_mpe(config)

    # mpe_model.initialize_model(mpe_dataloaders_dict['train']['UDataset'].dataset.data.shape[1])
    train_unlabeled_size = len(mpe_dataloaders_dict['train']['UDataset'].dataset.data)
 
    if config['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(mpe_model.net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['wd'])
    elif config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(mpe_model.net.parameters(), lr=config['lr'], weight_decay=config['wd'])
    elif config['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(mpe_model.net.parameters(), lr=config['lr'])

    scores_dict = {}
    all_scores_dict = {split: {'epochs': []} for split in ['train', 'val']}

    ## Train in the beginning for warm start
    if config['warm_start']:
        for epoch in range(config['warm_start_epochs']):
            scores_dict['train'] = mpe_model.warm_up_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'],
                                                       u_trainloader=mpe_dataloaders_dict['train']['UDataset'],
                                                       optimizer=optimizer, criterion=criterion, valid_loader=None)
            all_scores_dict['train']['epochs'].append(epoch)

            scores_dict['val'] = mpe_model.validate(epoch, p_validloader=mpe_dataloaders_dict['val']['PDataset'],
                                                u_validloader=mpe_dataloaders_dict['val']['UDataset'],
                                                criterion=criterion, threshold=0.5)
            all_scores_dict['val']['epochs'].append(epoch)

            if config['estimate_alpha']:
                mpe_model.alpha_estimate = mpe_model.estimate_alpha(p_holdoutloader=mpe_dataloaders_dict['holdout']['PDataset'],
                                                                    u_holdoutloader=mpe_dataloaders_dict['holdout']['UDataset'])
                mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))
            LOG.info(f"Warmup Epoch {epoch}: {scores_dict}")

            for split in all_scores_dict.keys():
                for score_type, score_value in scores_dict[split].items():
                    if score_type not in all_scores_dict[split]:
                        all_scores_dict[split][score_type] = []
                    all_scores_dict[split][score_type].append(score_value)

    best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = copy.deepcopy(mpe_model.state_dict())
    for epoch in range(config['epochs']):
        train_scores = mpe_model.train_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'],
                                                 u_trainloader=mpe_dataloaders_dict['train']['UDataset'],
                                                 optimizer=optimizer, criterion=criterion,
                                                 train_unlabeled_size=train_unlabeled_size)
        scores_dict['train'] = train_scores
        all_scores_dict['train']['epochs'].append(epoch + config['warm_start_epochs'])

        scores_dict['val'] = mpe_model.validate(epoch, p_validloader=mpe_dataloaders_dict['val']['PDataset'],
                                                u_validloader=mpe_dataloaders_dict['val']['UDataset'],
                                                criterion=criterion, threshold=0.5)
        all_scores_dict['val']['epochs'].append(epoch + config['warm_start_epochs'])

        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            best_scores_dict = copy.deepcopy(scores_dict)
            best_model_state = copy.deepcopy(mpe_model.state_dict())  
        for split in ['train', 'val']:
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        if config['estimate_alpha']:
            mpe_model.alpha_estimate = mpe_model.estimate_alpha(mpe_dataloaders_dict['holdout']['PDataset'], mpe_dataloaders_dict['holdout']['UDataset'])
            mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))
            
        LOG.info(f"Train Epoch {epoch}: {scores_dict}")
        
        # Add checkpointing code
        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch, 
                     "model_state": mpe_model.state_dict(),
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

    model = mpe_model
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
        best_scores_dict['test'] = model.validate(epoch, p_validloader=mpe_dataloaders_dict['test']['PDataset'],
                                                u_validloader=mpe_dataloaders_dict['test']['UDataset'],
                                                criterion=criterion, threshold=0.5)
                                                
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