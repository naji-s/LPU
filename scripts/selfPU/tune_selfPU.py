import json
import os
import datetime
import time
import tempfile

from matplotlib import pyplot as plt
import ray.train
import ray.tune
import ray.tune.schedulers
import torch

import LPU.scripts
import LPU.scripts.selfPU
import LPU.scripts.selfPU.run_selfPU
import LPU.models.selfPU.selfPU
import LPU.utils.dataset_utils
import LPU.utils.utils_general
import LPU.utils.ray_utils

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'selfPU'

def SET_DEFAULT_SEARCH_SPACE():
    default_search_space = {
        "random_state": None,
        "lr": .01,
        "weight_decay": ray.tune.loguniform(1e-6, 1e-3),
        "epochs": ray.tune.randint(100, 100 + 1),
        "weight": ray.tune.uniform(0.5, 2.0),
        "self_paced_start": ray.tune.randint(5, 20),
        "self_paced_stop": ray.tune.randint(5, 100),
        "self_paced_frequency": ray.tune.randint(5, 15),
        "self_paced_type": ray.tune.choice(["A", "B", "C"]),
        "ema_decay": ray.tune.loguniform(0.99, 0.999),
        "consistency": ray.tune.uniform(0.1, 0.5),
        "consistency_rampup": ray.tune.randint(200, 600),
        "top1": ray.tune.uniform(0.2, 0.6),
        "top2": ray.tune.uniform(0.4, 0.8),
        "gamma": ray.tune.loguniform(1e-3, 0.1),
        "num_p": ray.tune.randint(500, 2000),
        "batch_size": {
            "train": ray.tune.choice([64]),
            "test": ray.tune.choice([64]),
            "val": ray.tune.choice([64]),
            "holdout": ray.tune.choice([64])
        },
    }
    return default_search_space

def main(config=None, num_samples=100, max_num_epochs=200, gpus_per_trial=0, results_dir=None, random_state=None, tune=False):
    # setting the seed for the tuning
    if random_state is not None:
        LPU.utils.utils_general.set_seed(random_state)

    DEFAULT_SEARCH_SPACE = SET_DEFAULT_SEARCH_SPACE()
    if config is None:
        config = {}
    if tune:
        config = LPU.utils.utils_general.deep_update(DEFAULT_SEARCH_SPACE, config)
    else:
        config = LPU.utils.utils_general.deep_update(LPU.models.selfPU.selfPU.DEFAULT_CONFIG, config)

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
    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(data_config, target_transform=LPU.utils.dataset_utils.one_zero_to_minus_one_one,
                                                                       label_transform=LPU.utils.dataset_utils.one_zero_to_minus_one_one)

    result = ray.tune.run(
        ray.tune.with_parameters(LPU.scripts.selfPU.run_selfPU.train_model, dataloaders_dict=dataloaders_dict, with_ray=True),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        metric='val_overall_loss',
        scheduler=scheduler,
        mode='min',
        storage_path=results_dir,
        progress_reporter=reporter,
        keep_checkpoints_num=1,
        callbacks=[LPU.utils.ray_utils.PlotMetricsCallback(allow_list=['time_this_iter_s', 'epoch', 'val_overall_loss', 'val_y_APS', 'val_y_accuracy', 'val_y_auc'])],
        )

    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")

    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_model_checkpoint = torch.load(os.path.join(best_trial.checkpoint.path, "checkpoint.pt"))
    best_model = LPU.models.selfPU.selfPU.selfPU(config=best_model_checkpoint["config"])
    best_model.model1 = best_model.create_model().to(best_model_checkpoint["config"]['device'])
    best_model.model2 = best_model.create_model().to(best_model_checkpoint["config"]['device'])
    best_model.ema_model1 = best_model.create_model(ema = True).to(best_model_checkpoint["config"]['device'])
    best_model.ema_model2 = best_model.create_model(ema = True).to(best_model_checkpoint["config"]['device'])

    best_model.model1.load_state_dict(best_model_checkpoint["model1"])
    best_model.model2.load_state_dict(best_model_checkpoint["model2"])
    best_model.ema_model1.load_state_dict(best_model_checkpoint["ema_model1"])
    best_model.ema_model2.load_state_dict(best_model_checkpoint["ema_model2"])
    dataloader_test = best_model_checkpoint["dataloader_test"]
    criterion = LPU.scripts.selfPU.run_selfPU.get_criterion(best_model_checkpoint["config"])
    best_model_test_results = best_model.validate(dataloader_test, loss_fn=criterion, model=best_model.model1)
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
        "test_y_APS": final_results["test_y_APS"]},
    "Execution Time": execution_time,
    }
    # Storing results in a JSON file
    EXPERIMENT_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_save_dir = os.path.join(results_dir, MODEL_NAME, EXPERIMENT_DATETIME)

    # Ensure the directory exists
    os.makedirs(json_save_dir, exist_ok=True)
    json_save_path = os.path.join(json_save_dir, "best_trial_results.json")
    
    with open(json_save_path, "w") as json_file:
        json.dump(best_trial_report, json_file, indent=4, cls=LPU.utils.utils_general.CustomJSONEncoder)

    print(json.dumps(best_trial_report, indent=4, cls=LPU.utils.utils_general.CustomJSONEncoder))
    return best_trial_report

if __name__ == "__main__":
    args = LPU.utils.utils_general.tune_parse_args()
    main(num_samples=args.num_samples, max_num_epochs=args.max_num_epochs,
         gpus_per_trial=args.gpus_per_trial, results_dir=args.results_dir,
         random_state=args.random_state, tune=args.tune)