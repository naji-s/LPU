import json
import sys
import os
import datetime
import time

sys.path.append('LPU/external_libs/PU_learning')
sys.path.append('LPU/external_libs/PU_learning/data_helper')

from matplotlib import pyplot as plt
import ray.train
import ray.tune
import ray.tune.schedulers
import torch

import LPU.scripts.MPE.run_MPE
import LPU.utils.utils_general
import LPU.models.MPE.MPE
import LPU.utils.ray_utils

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'mpe'

def SET_DEFAULT_SEARCH_SPACE():
    default_search_space = {
        "random_state": ray.tune.randint(0, 1000),
        "lr": .01,
        "momentum": ray.tune.uniform(0.1, 0.9),
        "wd": ray.tune.loguniform(1e-6, 1e-2),
        "warm_start_epochs": ray.tune.choice([10, 20, 50]),
        "epochs": ray.tune.choice(range(100, 100 + 1)),
        "beta": ray.tune.uniform(0.3, 0.7),
        "train_method": ray.tune.choice(["TEDn"]),
        "net_type": ray.tune.choice(["LeNet"]),
        "sigmoid_loss": ray.tune.choice([True]),
        "estimate_alpha": ray.tune.choice([True]),
        "warm_start": ray.tune.choice([True]),
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
        config = LPU.utils.utils_general.deep_update(LPU.models.MPE.MPE.DEFAULT_CONFIG, config)

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
        "val_overall_loss", "val_l_auc", "val_l_accuracy", "val_l_APS"])
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    execution_start_time = time.time()
    dataloaders_dict = LPU.models.MPE.MPE.create_dataloaders_dict_mpe(data_config)

    experiment_name = f'train_{MODEL_NAME}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    EXPERIMENT_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_save_dir = os.path.join(results_dir, MODEL_NAME, EXPERIMENT_DATETIME)
    # Ensure the directory exists
    os.makedirs(json_save_dir, exist_ok=True)

    result = ray.tune.run(
        ray.tune.with_parameters(LPU.scripts.MPE.run_MPE.train_model, dataloaders_dict=dataloaders_dict),
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
        callbacks=[LPU.utils.ray_utils.PlotMetricsCallback(allow_list=['time_this_iter_s', 'epoch', 'val_overall_loss', 'val_y_APS', 'val_y_accuracy', 'val_y_auc'], json_save_dir=json_save_dir)],
    )
    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")

    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_model_checkpoint = torch.load(os.path.join(best_trial.checkpoint.path, "checkpoint.pt"))
    best_model = LPU.models.MPE.MPE.MPE(config=best_model_checkpoint["config"])
    best_model.load_state_dict(best_model_checkpoint["model_state"])
    
    criterion = torch.nn.CrossEntropyLoss()

    best_model_test_results = best_model.validate(p_validloader=dataloaders_dict['test']['PDataset'],
                                                u_validloader=dataloaders_dict['test']['UDataset'], criterion=criterion, threshold=0.5, epoch=-1)
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