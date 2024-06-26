import json
import os
import datetime
import time
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import ray.train
import ray.tune
import ray.tune.schedulers
import torch

import LPU.scripts
import LPU.scripts.SARPU
import LPU.scripts.SARPU.run_SARPU
import LPU.models.SARPU.SARPU
import LPU.utils.dataset_utils
import LPU.utils.utils_general
import LPU.utils.ray_utils

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'SARPU'

def SET_DEFAULT_SEARCH_SPACE():
    default_search_space = {
        "random_state": None,
        "num_epochs": ray.tune.choice(range(100, 100 + 1)),
        "SAR_PU_classification_model": ray.tune.choice(['logistic']),
        "svm_params": {
            "tol": ray.tune.loguniform(1e-5, 1e-3),
            "C": ray.tune.loguniform(0.1, 10),
            "kernel": ray.tune.choice(['linear', 'rbf']),
            "degree": ray.tune.randint(2, 5),
            "gamma": ray.tune.loguniform(1e-4, 1),
            "max_iter": ray.tune.choice([-1, 1000, 2000, 5000]),
        },
        "logistic_params": {
            "penalty": ray.tune.choice(['l2']),
            "tol": ray.tune.loguniform(1e-5, 1e-3),
            "C": ray.tune.loguniform(0.1, 10),
            "fit_intercept": ray.tune.choice([True, False]),
            "solver": ray.tune.choice(['lbfgs']),
            "max_iter": ray.tune.choice([1000, 2000, 5000]),
        },
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
        config = LPU.utils.utils_general.deep_update(LPU.models.SARPU.SARPU.DEFAULT_CONFIG, config)

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

    experiment_name = f'train_{MODEL_NAME}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    EXPERIMENT_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_save_dir = os.path.join(results_dir, MODEL_NAME, EXPERIMENT_DATETIME)
    # Ensure the directory exists
    os.makedirs(json_save_dir, exist_ok=True)

    result = ray.tune.run(
        ray.tune.with_parameters(LPU.scripts.SARPU.run_SARPU.train_model, dataloaders_dict=dataloaders_dict, with_ray=True),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        metric='val_overall_loss',
        scheduler=scheduler,
        mode='min',
        storage_path=results_dir,
        progress_reporter=reporter,
        keep_checkpoints_num=1,
        name=experiment_name,
        callbacks=[LPU.utils.ray_utils.PlotMetricsCallback(allow_list=['time_this_iter_s', 'val_overall_loss', 'val_y_APS', 'val_y_accuracy', 'val_y_auc'], json_save_dir=json_save_dir)],
        )

    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")
    best_trial = result.get_best_trial("val_overall_loss", "min", "last")

    best_model_checkpoint = torch.load(os.path.join(best_trial.checkpoint.path, "checkpoint.pt"))
    best_model = LPU.models.SARPU.SARPU.SARPU(best_model_checkpoint["config"], 
                                              training_size=len(dataloaders_dict['train'].dataset))
    best_model.load_state_dict(best_model_checkpoint["model_state"])
    
    # since SARPU uses sklearn for the propensity & classification model, 
    # we need to retrain the model on the whole training set, as the weights
    # are not stored in the state of the our wrapper model around SARPU training
    # algorithm from its original repo
    dataloader = dataloaders_dict['train']    
    X = []
    l = []
    y = []
    # put all the data in one list
    for data in dataloader:
        X_batch, l_batch, y_batch, _ = data
        X.append(X_batch.numpy())
        l.append(l_batch.numpy())
        y.append(y_batch.numpy())
    # concatenate the data
    X = np.concatenate(X)
    l = np.concatenate(l)
    y = np.concatenate(y)
            
    propensity_attributes = np.ones(X.shape[-1]).astype(int)
    best_model.classification_model, best_model.propensity_model, best_model.results = LPU.external_libs.SAR_PU.sarpu.sarpu.pu_learning.pu_learn_sar_em(
            X, l, classification_model=best_model.classification_model, propensity_attributes=propensity_attributes, max_its=best_model.max_iter)
    
    best_model_test_results = best_model.validate(dataloaders_dict['test'], loss_fn=best_model.loss_fn)
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
