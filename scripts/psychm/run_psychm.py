import json
import torch
import lpu.constants
import lpu.datasets.dataset_utils
import lpu.models.geometric.psychmGGPC
import lpu.utils.plot_utils
import lpu.utils.utils_general

torch.set_default_dtype(lpu.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    "inducing_points_size": 32,
    "learning_rate": 0.01,
    "num_epochs": 59,
    "device": "cpu",
    "epoch_block": 1,
    "intrinsic_kernel_params": {
        "normed": False,
        "kernel_type": "laplacian",
        "heat_temp": 0.01,
        "noise_factor": 0.0,
        "amplitude": 0.5,
        "n_neighbor": 5,
        "lengthscale": 0.3,
        "neighbor_mode": "distance",
        "power_factor": 1,
        "invert_M_first": False,
        "normalize": False
    },
    "dataset_name": "animal_no_animal",
    "dataset_kind": "LPU",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "ratios": {
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
    base_config = lpu.utils.utils_general.deep_update(DEFAULT_CONFIG, config)

    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)

    inducing_points_size = base_config['inducing_points_size']
    dataloaders_dict = lpu.utils.dataset_utils.create_dataloaders_dict(base_config)
    inducing_points_initial_vals = lpu.utils.dataset_utils.initialize_inducing_points(
        dataloaders_dict['train'], inducing_points_size)
    psychmGVGP_model = lpu.models.geometric.psychmGGPC.PsychMGP(
        base_config,
        inducing_points_initial_vals=inducing_points_initial_vals,
        training_size=len(dataloaders_dict['train'].dataset),
        num_features=inducing_points_initial_vals.shape[-1]
    )

    learning_rate = base_config['learning_rate']
    num_epochs = base_config['num_epochs']
    epoch_block = base_config['epoch_block']
    optimizer = torch.optim.Adam([{
        'params': psychmGVGP_model.parameters(),
        'lr': learning_rate
    }])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    all_scores_dict = {split: {'epochs': []} for split in ['train', 'val']}
    scores_dict = {split: {} for split in ['train', 'val']}

    best_val_loss = float('inf')
    best_epoch = -1
    # breakpoint()
    for epoch in range(num_epochs):
        scores_dict['train'] = psychmGVGP_model.train_one_epoch(dataloaders_dict['train'], optimizer)
        all_scores_dict['train']['epochs'].append(epoch)

        scores_dict['val'] = psychmGVGP_model.validate(dataloaders_dict['val'], loss_fn=psychmGVGP_model.loss_fn)
        all_scores_dict['val']['epochs'].append(epoch)
        scheduler.step(scores_dict['val']['overall_loss'])

        for split in ['train', 'val']:
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        if scores_dict['val']['overall_loss'] < best_val_loss:
            best_val_loss = scores_dict['val']['overall_loss']
            best_epoch = epoch

        LOG.info(f"Epoch {epoch}: {json.dumps(scores_dict, indent=2)}")

    # Evaluate on the test set after training
    scores_dict['test'] = psychmGVGP_model.validate(dataloaders_dict['test'], loss_fn=psychmGVGP_model.loss_fn)


    # Flatten scores_dict
    flattened_scores = lpu.utils.utils_general.flatten_dict(scores_dict)
    filtered_scores_dict = {}
    for key, value in flattened_scores.items():
        if 'train' in key or 'val' in key or 'test' in key:
            if 'epochs' not in key:
                filtered_scores_dict[key] = value
    print("Reporting Metrics: ", filtered_scores_dict)  # Debug print to check keys

    # Report metrics if executed under Ray Tune
    if RAY_AVAILABLE and (ray.util.client.ray.is_connected() or ray.is_initialized()):
        ray.train.report(filtered_scores_dict)
    else:
        return all_scores_dict, best_epoch

if __name__ == "__main__":
    scores, best_epoch = train_model()
    lpu.utils.plot_utils.plot_scores(scores, loss_type='overall_loss', best_epoch=best_epoch)
