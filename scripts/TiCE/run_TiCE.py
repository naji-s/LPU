import copy

import torch

import LPU.constants
import LPU.utils.dataset_utils
import LPU.models.TiCE.TiCE
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

def train_model(config=None, dataloaders_dict=None):
    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(DEFAULT_CONFIG, config)

    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
        # setting the seed for the training
        LPU.utils.utils_general.set_seed(random_state)



    inducing_points_size = config['inducing_points_size']
    if dataloaders_dict is None:
        dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    



    inducing_points_size = config['inducing_points_size']
    inducing_points_initial_vals = LPU.utils.dataset_utils.initialize_inducing_points(
        dataloaders_dict['train'], inducing_points_size)
    tice_model = LPU.models.TiCE.TiCE.TiCE(
        config,
        inducing_points_initial_vals=inducing_points_initial_vals,
        training_size=len(dataloaders_dict['train'].dataset),
        num_features=inducing_points_initial_vals.shape[-1]
    )

    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    epoch_block = config['epoch_block']
    optimizer = torch.optim.Adam([{
        'params': tice_model.parameters(),
        'lr': learning_rate
    }])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}

    best_val_loss = float('inf')
    best_epoch = -1
    best_scores_dict = None
    best_model_state = copy.deepcopy(tice_model.state_dict())

    tice_model.set_C(dataloaders_dict['holdout'])
    for epoch in range(num_epochs):
        scores_dict = {split: {} for split in dataloaders_dict.keys()}
        scores_dict_item = tice_model.train_one_epoch(optimizer=optimizer, dataloader=dataloaders_dict['train'],
                                                     holdout_dataloader=dataloaders_dict['holdout'])
        scores_dict['train'].update(scores_dict_item)
        all_scores_dict['train']['epochs'].append(epoch)

        if epoch % epoch_block == 0:
            scores_dict_item = tice_model.validate(dataloaders_dict['val'], loss_fn=tice_model.loss_fn, model=tice_model.gp_model)
            scores_dict['val'].update(scores_dict_item)
            all_scores_dict['val']['epochs'].append(epoch)

            scheduler.step(scores_dict['val']['overall_loss'])
            # Update best validation loss and epoch
            if scores_dict['val']['overall_loss'] < best_val_loss:
                best_val_loss = scores_dict['val']['overall_loss']
                best_epoch = epoch
                best_scores_dict = copy.deepcopy(scores_dict)
                best_model_state = copy.deepcopy(tice_model.state_dict())
        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Epoch {epoch}: {scores_dict}")
        # Check current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        LOG.info(f"Current learning rate: {current_lr}")
        # Stop if the learning rate is too low
        if current_lr <= config['stop_learning_lr']:
            print("Learning rate below threshold, stopping training.")
            break

        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
                        ray.train.report({
                            'val_overall_loss': scores_dict['val']['overall_loss'],
                            'epoch': epoch,
                            'learning_rate': current_lr})        


    LOG.info(f"Best epoch: {best_epoch}, Best validation overall_loss: {best_val_loss:.5f}")

    model = tice_model
    # Evaluate on the test set with the best model based on the validation set
    model.load_state_dict(best_model_state)

    best_scores_dict['test'] = model.validate(dataloaders_dict['test'], loss_fn=model.loss_fn, model=model.gp_model)

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
    results, best_epoch = train_model()
    LPU.utils.plot_utils.plot_scores(results, best_epoch=best_epoch, loss_type='overall_loss')
