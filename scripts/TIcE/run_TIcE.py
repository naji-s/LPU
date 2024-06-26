import copy
import json
import tempfile
import os

import torch

import LPU.constants
import LPU.utils.dataset_utils
import LPU.models.TIcE.TIcE
import LPU.utils.plot_utils
import LPU.utils.utils_general

torch.set_default_dtype(LPU.constants.DTYPE)

USE_DEFAULT_CONFIG = False


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
    config = LPU.utils.utils_general.deep_update(LPU.models.TIcE.TIcE.DEFAULT_CONFIG, config)

    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
        # setting the seed for the training
        LPU.utils.utils_general.set_seed(random_state)



    inducing_points_size = config['inducing_points_size']
    if dataloaders_dict is None:
        dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    inducing_points_initial_vals = LPU.utils.dataset_utils.initialize_inducing_points(
        dataloaders_dict['train'], inducing_points_size)
    TIcE_model = LPU.models.TIcE.TIcE.TIcE(
        config,
        inducing_points_initial_vals=inducing_points_initial_vals,
        training_size=len(dataloaders_dict['train'].dataset),
    )

    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    optimizer = torch.optim.Adam([{
        'params': TIcE_model.parameters(),
        'lr': learning_rate
    }])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    all_scores_dict = {split: {'epochs': []} for split in ['train', 'val']}
    scores_dict = {split: {} for split in ['train', 'val']}
    best_val_loss = float('inf')
    best_epoch = -1
    best_scores_dict = None
    best_model_state = None
    try:
        TIcE_model.set_C(dataloaders_dict['holdout'])
    except ZeroDivisionError:
        # based on what TIcE initially sets C to be, 
        # in cases where low_c fails in LPU/external_libs/SAR_PU/lib/tice/tice/tice.py
        # due to division by zero
        TIcE_model.C.fill_(0.5)
    for epoch in range(num_epochs):
        scores_dict['train'] = TIcE_model.train_one_epoch(optimizer=optimizer, dataloader=dataloaders_dict['train'], 
                                                          holdout_dataloader=dataloaders_dict['holdout'])
        all_scores_dict['train']['epochs'].append(epoch)

        scores_dict['val'] = TIcE_model.validate(dataloaders_dict['val'], loss_fn=TIcE_model.loss_fn, model=TIcE_model.gp_model)
        all_scores_dict['val']['epochs'].append(epoch)
        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            best_scores_dict = copy.deepcopy(scores_dict)
            best_model_state = copy.deepcopy(TIcE_model.state_dict())

        scheduler.step(scores_dict['val']['overall_loss'])
        for split in ['train', 'val']:
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)


        LOG.info(f"Epoch {epoch}: {json.dumps(scores_dict, indent=2)}")
        current_lr = optimizer.param_groups[0]['lr']
        LOG.info(f"Current learning rate: {current_lr}")

        # Add checkpointing code
        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {"epoch": epoch,
                     "model_state": TIcE_model.state_dict(),
                     "config": config,},
                    os.path.join(tempdir, "checkpoint.pt"),
                )
                ray.train.report(metrics={
                        'val_overall_loss': scores_dict['val']['overall_loss'],
                        'val_y_auc': scores_dict['val']['y_auc'],
                        'val_y_accuracy': scores_dict['val']['y_accuracy'],
                        'val_y_APS': scores_dict['val']['y_APS'],
                        'C_estimate': TIcE_model.C.item(),
                        'epoch': epoch,
                        'learning_rate': current_lr}, checkpoint=ray.train.Checkpoint.from_directory(tempdir))
                
        # Stop if the learning rate is too low
        if current_lr <= config['stop_learning_lr']:
            print("Learning rate below threshold, stopping training.")
            break

    LOG.info(f"Best epoch: {best_epoch}, Best validation overall_loss: {best_val_loss:.5f}")

    model = TIcE_model
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
        best_scores_dict['test'] = model.validate(dataloaders_dict['test'], loss_fn=model.loss_fn, model=model.gp_model)
        flattened_scores = LPU.utils.utils_general.flatten_dict(best_scores_dict)
        filtered_scores_dict = {}
        for key, value in flattened_scores.items():
            if 'train' in key or 'val' in key or 'test' in key:
                if 'epochs' not in key:
                    filtered_scores_dict[key] = value
        LOG.info(f"Final test scores: {best_scores_dict['test']}")
        return all_scores_dict, best_epoch

if __name__ == "__main__":
    results, best_epoch = train_model()
    LPU.utils.plot_utils.plot_scores(results, best_epoch=best_epoch, loss_type='overall_loss')