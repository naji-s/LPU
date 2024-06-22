import json
import os
import datetime
import time
import tempfile

import ray.train
import ray.tune
import ray.tune.schedulers
import torch

import LPU.models.distPU.distPU
import LPU.scripts
import LPU.scripts.distPU
import LPU.scripts.distPU.run_distPU
import LPU.models.distPU
import LPU.utils.dataset_utils
import LPU.utils.utils_general


LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'distPU'

def main(num_samples=100, max_num_epochs=200, gpus_per_trial=0, results_dir=None, random_state=None):
    if random_state is None:
        LOG.warning("seed_num is None. Setting it to 0.")
        random_state = 0
    search_space = {
        "random_state": random_state,
        "warm_up_lr": .01,
        "lr": .01,
        # "warm_up_weight_decay": ray.tune.loguniform(1e-6, 1e-2),
        # "weight_decay": ray.tune.loguniform(1e-6, 1e-2),        
        "pu_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "warm_up_epochs": ray.tune.choice([10, 20, 50]),
        "co_entropy": ray.tune.uniform(0.001, 0.01),
        "optimizer": ray.tune.choice(['adam']),
        # "schedular": ray.tune.choice(['cos-ann', 'step']),
        "entropy": ray.tune.uniform(0.1, 1.0),
        "co_mu": ray.tune.uniform(0.001, 0.01),
        "co_mix_entropy": ray.tune.uniform(0.01, 0.1),
        "co_mixup": ray.tune.uniform(2.0, 10.0),
        "batch_size": {
            "train": ray.tune.choice([64]),
            "test": ray.tune.choice([64]),
            "val": ray.tune.choice([64]),
            "holdout": ray.tune.choice([64])
        },
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
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS"])
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    execution_start_time = time.time()
    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(data_config)

    result = ray.tune.run(
        ray.tune.with_parameters(LPU.scripts.distPU.run_distPU.train_model, dataloaders_dict=dataloaders_dict, with_ray=True),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        scheduler=scheduler,
        mode='min',
        local_dir=results_dir,
        progress_reporter=reporter,
        keep_checkpoints_num=1
        )
    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")

    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_model_checkpoint = torch.load(os.path.join(best_trial.checkpoint.path, "checkpoint.pt"))
    best_model = LPU.models.distPU.distPU.distPU(config=best_model_checkpoint["config"],
                                          dim=torch.flatten(dataloaders_dict['train'].dataset.X, 1).shape[1])
    best_model.load_state_dict(best_model_checkpoint["model_state"])

    best_model_test_results = best_model.validate(dataloaders_dict['test'], loss_fn=LPU.models.distPU.distPU.create_loss(best_model_checkpoint["config"]), model=best_model.model)
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