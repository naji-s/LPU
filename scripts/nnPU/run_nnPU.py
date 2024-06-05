import logging
import unittest.mock
import types
import numpy as np
import torch.optim
import lpu.constants
import lpu.models.uPU
import lpu.utils.dataset_utils
import lpu.utils.dataset_utils
import lpu.utils.utils_general
import lpu.models.nnPU
import lpu.utils.plot_utils
import lpu.external_libs.nnPUSB
import lpu.external_libs.nnPUSB.nnPU_loss
import lpu.external_libs.nnPUSB.dataset

torch.set_default_dtype(lpu.constants.DTYPE)

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
    "out": "lpu/scripts/nnPU/checkpoints",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "dataset_kind": "LPU",
    "batch_size": {
        "train": 64,
        "test": None,
        "val": None,
        "holdout": None
    },
    "ratios": {
        "test": 0.25,
        "val": 0.2,
        "holdout": 0.05,
        "train": 0.5
    },
}


LOG = lpu.utils.utils_general.configure_logger(__name__)

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
    config = lpu.utils.utils_general.deep_update(DEFAULT_CONFIG, config)
    gamma = config.get('gamma')
    beta = config.get('beta')
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)

    dataloaders_dict = lpu.utils.dataset_utils.create_dataloaders_dict(config, target_transform=lpu.utils.dataset_utils.one_zero_to_minus_one_one,
                                                                       label_transform=lpu.utils.dataset_utils.one_zero_to_minus_one_one)
    dim = dataloaders_dict['train'].dataset.X.shape[-1]
    nnPU_model = lpu.models.nnPU.nnPU(config=config, dim=dim)
    nnPU_model.set_C(dataloaders_dict['holdout'])

    optimizer = torch.optim.Adam([{
        'params': nnPU_model.parameters(),
        'lr': config.get('learning_rate', DEFAULT_CONFIG.get('learning_rate', None) if USE_DEFAULT_CONFIG else None)
    }])
    device = config.get('device', 'cpu')
    loss_func = lpu.external_libs.nnPUSB.nnPU_loss.nnPUloss(prior=nnPU_model.prior,
                                         loss=lpu.models.uPU.select_loss('sigmoid'),
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
    flattened_scores = lpu.utils.utils_general.flatten_dict(scores_dict)
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
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/nnPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    # args = types.SimpleNamespace(**config)
    # with unittest.mock.patch('lpu.external_libs.nnPUSB.args.device', return_value=args.device):
    #     import lpu.external_libs.nnPUSB.train
    #     import lpu.external_libs.nnPUSB.model
    results, best_epoch = train_model()
    lpu.utils.plot_utils.plot_scores(results, best_epoch=best_epoch, loss_type='overall_loss')