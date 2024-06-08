import logging
import unittest.mock
import types
import numpy as np
import torch.optim
import LPU.constants
import LPU.models.uPU
import LPU.utils.dataset_utils
import LPU.utils.dataset_utils
import LPU.utils.utils_general
import LPU.models.nnPU
import LPU.utils.plot_utils
import LPU.external_libs.nnPUSB
import LPU.external_libs.nnPUSB.nnPU_loss
import LPU.external_libs.nnPUSB.dataset

torch.set_default_dtype(LPU.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    "device": "cpu",
    "preset": "figure1",
    "dataset_name": "animal_no_animal",
    "numpy_dtype": "float64",
    "epoch": 100,
    "beta": 0.0,
    "gamma": 1.0,
    "learning_rate": 0.001,
    "loss": "sigmoid",
    "model": "mlp",
    "out": "LPU/scripts/nnPU/checkpoints",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "dataset_kind": "LPU",
    "batch_size": {
        "train": 64,
        "test": None,
        "val": None,
        "holdout": None
    },
    'ratios': 
    {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.25,
        'val': 0.1,
        'holdout': .0,
        'train': .65, 
    },

}


LOG = LPU.utils.utils_general.configure_logger(__name__)

# Optional dynamic import for Ray
try:
    import ray.util.client
    import ray.train
    RAY_AVAILABLE = True
except ImportError:
    LOG.warning("Ray is not available. Please install Ray to enable distributed training.")
    RAY_AVAILABLE = False



def train_model(config=None):
    if config is None:
        config = {}
    # Load the base configuration
    config = LPU.utils.utils_general.deep_update(DEFAULT_CONFIG, config)
    gamma = config.get('gamma')
    beta = config.get('beta')
    LPU.utils.utils_general.set_seed(LPU.constants.RANDOM_STATE)

    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config, target_transform=LPU.utils.dataset_utils.one_zero_to_minus_one_one,
                                                                       label_transform=LPU.utils.dataset_utils.one_zero_to_minus_one_one)
    dim = dataloaders_dict['train'].dataset.X.shape[-1]
    nnPU_model = LPU.models.nnPU.nnPU(config=config, dim=dim)
    nnPU_model.set_C(dataloaders_dict['holdout'])

    optimizer = torch.optim.Adam([{
        'params': nnPU_model.parameters(),
        'lr': config.get('learning_rate', DEFAULT_CONFIG.get('learning_rate', None) if USE_DEFAULT_CONFIG else None)
    }])
    device = config.get('device', 'cpu')
    loss_func = LPU.external_libs.nnPUSB.nnPU_loss.nnPUloss(prior=nnPU_model.prior,
                                         loss=LPU.models.uPU.select_loss('sigmoid'),
                                         gamma=gamma,
                                         beta=beta)
    num_epochs = config.get('epoch', DEFAULT_CONFIG.get('epoch', None) if USE_DEFAULT_CONFIG else None)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}
    scores_dict = {split: {} for split in dataloaders_dict.keys()}
    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(num_epochs):
        scores_dict['train'] = nnPU_model.train_one_epoch(dataloader=dataloaders_dict['train'], optimizer=optimizer, loss_fn=loss_func, device=device)
        all_scores_dict['train']['epochs'].append(epoch)

        scores_dict['val'] = nnPU_model.validate(dataloaders_dict['val'], model=nnPU_model.model, loss_fn=loss_func)
        all_scores_dict['val']['epochs'].append(epoch)

        for split in ['train', 'val']:
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Epoch {epoch}: {scores_dict}")

        # Update best validation loss and epoch
        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch
            # best_scores_dict = copy.deepcopy(scores_dict)

    scores_dict['test'] = nnPU_model.validate(dataloaders_dict['test'], loss_fn=loss_func, model=nnPU_model.model)
    # Flatten scores_dict
    flattened_scores = LPU.utils.utils_general.flatten_dict(scores_dict)
    filtered_scores_dict = {}
    for key, value in flattened_scores.items():
        if 'train' in key or 'val' in key or 'test' in key:
            if 'epochs' not in key:
                filtered_scores_dict[key] = value

    print("Reporting Metrics: ", filtered_scores_dict)  # Debug print to check keys
    LOG.info(f"Final test error: {scores_dict['test']}")

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