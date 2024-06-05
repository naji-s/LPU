import copy

import torch

import lpu.constants
import lpu.utils.dataset_utils
import lpu.models.tice
import lpu.utils.plot_utils
import lpu.utils.utils_general

torch.set_default_dtype(lpu.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    # for VGP:
    "inducing_points_size": 32,
    "learning_rate": 0.01,
    "num_epochs": 50,
    "device": "cpu",
    "set_C_every_epoch": False,
    "epoch_block": 1, # Perform validation every EPOCH_BLOCK iterations
    "kernel_mode": 2,
    "intrinsic_kernel_params": {
        "normed": False,
        "kernel_type": "laplacian",
        "heat_temp": 0.01,
        "noise_factor": 0.0,
        "amplitude": 0.,
        "n_neighbor": 5,
        "lengthscale": 0.3,
        "neighbor_mode": "distance",
        "power_factor": 1,
        "invert_M_first": False,
        "normalize": False
        },
    "out": None,
    "folds": None,
    "delta": None,
    "max-bepp": 5,
    "maxSplits": 500,
    "promis": False,
    "delimiter": ',',
    "minT": 10,
    "nbIts": 2,
    "dataset_name": "animal_no_animal",  # fashionMNIST
    "dataset_kind": "LPU",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "ratios": {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        "test": 0.25,
        "val": 0.2,
        "holdout": 0.05,
        "train": 0.5
    },
    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    }
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

    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)

    inducing_points_size = config['inducing_points_size']
    dataloaders_dict = lpu.utils.dataset_utils.create_dataloaders_dict(config)
    inducing_points_initial_vals = lpu.utils.dataset_utils.initialize_inducing_points(
        dataloaders_dict['train'], inducing_points_size)
    tice_model = lpu.models.tice.Tice(
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}

    best_val_loss = float('inf')
    best_epoch = -1

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

        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Epoch {epoch}: {scores_dict}")

    scores_dict['test'] = tice_model.validate(dataloaders_dict['test'], loss_fn=tice_model.loss_fn, model=tice_model.gp_model)

    # Flatten scores_dict
    flattened_scores = lpu.utils.utils_general.flatten_dict(scores_dict)
    filtered_scores_dict = {}
    for key, value in flattened_scores.items():
        if 'train' in key or 'val' in key or 'test' in key:
            if 'epochs' not in key:
                filtered_scores_dict[key] = value
    LOG.info(f"Final test error: {scores_dict['test']}")

    # Report metrics if executed under Ray Tune
    if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
        ray.train.report(filtered_scores_dict)
    else:
        return all_scores_dict, best_epoch

if __name__ == "__main__":
    results, best_epoch = train_model()
    lpu.utils.plot_utils.plot_scores(results, best_epoch=best_epoch, loss_type='overall_loss')
