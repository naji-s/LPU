import copy
import logging
import unittest.mock
import types
import numpy as np
import torch.optim
import LPU.constants
import LPU.models.uPU.uPU
import LPU.utils.dataset_utils
import LPU.utils.dataset_utils
import LPU.utils.utils_general
import LPU.models.nnPU.nnPU
import LPU.utils.plot_utils
import LPU.external_libs.nnPUSB
import LPU.external_libs.nnPUSB.nnPU_loss
import LPU.external_libs.nnPUSB.dataset

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

    if config['set_seed']:
        seed = config.get('random_state', LPU.constants.RANDOM_STATE)
        LPU.utils.utils_general.set_seed(seed)


    if dataloaders_dict is None:
        dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    

    gamma = config.get('gamma')
    beta = config.get('beta')


    dim = dataloaders_dict['train'].dataset.X.shape[-1]
    nnPU_model = LPU.models.nnPU.nnPU.nnPU(config=config, dim=dim)
    nnPU_model.set_C(dataloaders_dict['train'])

    optimizer = torch.optim.Adam([{
        'params': nnPU_model.parameters(),
        'lr': config.get('learning_rate', DEFAULT_CONFIG.get('learning_rate', None) if USE_DEFAULT_CONFIG else None)
    }])
    device = config.get('device', 'cpu')
    loss_fn = LPU.external_libs.nnPUSB.nnPU_loss.nnPUloss(prior=nnPU_model.prior,
                                         loss=LPU.models.uPU.uPU.select_loss('sigmoid'),
                                         gamma=gamma,
                                         beta=beta)
    num_epochs = config.get('epoch', DEFAULT_CONFIG.get('epoch', None) if USE_DEFAULT_CONFIG else None)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}
    scores_dict = {split: {} for split in dataloaders_dict.keys()}
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(nnPU_model.state_dict())
    best_epoch = -1
    for epoch in range(num_epochs):
        scores_dict['train'] = nnPU_model.train_one_epoch(dataloader=dataloaders_dict['train'], optimizer=optimizer, loss_fn=loss_fn, device=device)
        all_scores_dict['train']['epochs'].append(epoch)

        scores_dict['val'] = nnPU_model.validate(dataloaders_dict['val'], model=nnPU_model.model, loss_fn=loss_fn)
        all_scores_dict['val']['epochs'].append(epoch)

        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Epoch {epoch}: {scores_dict}")

        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            best_scores_dict = copy.deepcopy(scores_dict)
            best_model_state = copy.deepcopy(nnPU_model.state_dict())
        # Report metrics if executed under Ray Tune
        if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
                        ray.train.report({
                            'val_overall_loss': scores_dict['val']['overall_loss'],
                            'epoch': epoch,
                            })        

    LOG.info(f"Best epoch: {best_epoch}, Best validation overall_loss: {best_val_loss:.5f}")

    model = nnPU_model
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
    yaml_file_path = '/Users/naji/phd_codebase/LPU/configs/nnPU_config.yaml'
    config = LPU.utils.utils_general.load_and_process_config(yaml_file_path)
    # args = types.SimpleNamespace(**config)
    # with unittest.mock.patch('LPU.external_libs.nnPUSB.args.device', return_value=args.device):
    #     import LPU.external_libs.nnPUSB.train
    #     import LPU.external_libs.nnPUSB.model
    results, best_epoch = train_model()
    LPU.utils.plot_utils.plot_scores(results, best_epoch=best_epoch, loss_type='overall_loss')