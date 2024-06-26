import json
import os
import datetime
import time

import ray.train
import ray.tune
import ray.tune.schedulers
import torch

import LPU.scripts.Elkan.run_Elkan
import LPU.models.geometric.Elkan.Elkan
import LPU.utils.dataset_utils
import LPU.utils.utils_general

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'Elkan'

def main(num_samples=100, max_num_epochs=200, gpus_per_trial=0, results_dir=None, random_state=None):
    # setting the seed for the tuning
    if random_state is not None:
        LPU.utils.utils_general.set_seed(random_state)

    # Configuration for hyperparameters to be tuned
    search_space = {
        # making sure the model training is not gonna set the seed 
        # since we potentially might want to set the seed for the tuning
		"random_state": ray.tune.randint(0, 1000),
        "inducing_points_size": ray.tune.choice([64]),
        "num_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "learning_rate": 0.01,
        "intrinsic_kernel_params": {
            "normed": ray.tune.choice([False]),
            "kernel_type": ray.tune.choice(["laplacian"]),
            "noise_factor": ray.tune.uniform(1e-4, 1e-1),
            "amplitude": ray.tune.loguniform(0.01, 100.0),
            "n_neighbor": ray.tune.choice([5]),
            "lengthscale": ray.tune.loguniform(0.01, 100.0),
            "neighbor_mode": ray.tune.choice(["connectivity"]),
            "power_factor": ray.tune.uniform(0.1, 2.0),
        },
        # "batch_size": {
        #     "train": ray.tune.choice([64]),
        #     "test": ray.tune.choice([64]),
        #     "val": ray.tune.choice([64]),
        #     "holdout": ray.tune.choice([64])
        # },
    }
    data_config = {
        "dataset_name": "animal_no_animal",  # fashionMNIST
        "dataset_kind": "LPU",
        "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
        "device": "cpu",
        'ratios':
        {
            # *** NOTE ***
            # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
            # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
            'test': 0.4,
            'val': 0.05,
            'holdout': .05,
            'train': .5,
        },
        "batch_size": {
            "train": 64,
            "test": 64,
            "val": 64,
            "holdout": 64
        }
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS",
        "test_overall_loss", "test_y_auc", "test_y_accuracy", "test_y_APS"])

    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    execution_start_time = time.time()
    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(data_config)

    result = ray.tune.run(
        ray.tune.with_parameters(LPU.scripts.Elkan.run_Elkan.train_model, dataloaders_dict=dataloaders_dict, with_ray=True),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        scheduler=scheduler,
        mode='min',
        storage_path=results_dir,
        progress_reporter=reporter,
        keep_checkpoints_num=1
        )
    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")
    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_model_checkpoint = torch.load(os.path.join(best_trial.checkpoint.path, "checkpoint.pt"))
    best_model = LPU.models.geometric.Elkan.Elkan.Elkan(config=best_model_checkpoint["config"],
                                           inducing_points_initial_vals=best_model_checkpoint["model_state"]["inducing_points"],
                                           training_size=len(dataloaders_dict['train'].dataset))
    best_model.load_state_dict(best_model_checkpoint["model_state"])

    best_model_test_results = best_model.validate(dataloaders_dict['test'], loss_fn=best_model.loss_fn, model=best_model.gp_model)
    final_epoch = best_trial.last_result["training_iteration"]
    final_results = best_trial.last_result.copy()
    for key in best_trial.last_result:
        if 'val_' in key:
            final_results[key.replace('val', 'test')] = best_model_test_results['_'.join(key.split('_')[1:])]
    best_trial_report = {
        "Best trial config": best_trial.config,
        "Best trial final validation loss": final_results["val_overall_loss"],
        "Best trial final test scores": {
            "test_overall_loss": final_results["test_overall_loss"],
            "test_y_auc": final_results["test_y_auc"],
            "test_y_accuracy": final_results["test_y_accuracy"],
            "test_y_APS": final_results["test_y_APS"]
        },
        "Execution Time": execution_time,
        "Final epoch": final_epoch,
    }
    # Storing results in a JSON file
    EXPERIMENT_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_save_dir = os.path.join(results_dir, MODEL_NAME, EXPERIMENT_DATETIME)

    # Ensure the directory exists
    os.makedirs(json_save_dir, exist_ok=True)
    json_save_path = os.path.join(json_save_dir, "best_trial_results.json")
    
    with open(json_save_path, "w") as json_file:
        json.dump(best_trial_report, json_file, indent=4)

    print(json.dumps(best_trial_report, indent=4))
    return best_trial_report

if __name__ == "__main__":
    args = LPU.utils.utils_general.tune_parse_args()
    main(args.num_samples, args.max_num_epochs, args.gpus_per_trial, args.results_dir, args.random_state)