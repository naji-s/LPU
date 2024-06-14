import json
import sys
import os
import datetime
import time

sys.path.append('LPU/external_libs/PU_learning')
sys.path.append('LPU/external_libs/PU_learning/data_helper')

import ray.train
import ray.tune
import ray.tune.schedulers

import LPU.scripts.mpe.run_mpe
import LPU.utils.utils_general

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'mpe'


def main(num_samples=100, max_num_epochs=200, gpus_per_trial=0, results_dir=None, random_state=None):
    # Configuration for hyperparameters to be tuned
    if random_state is None:
        LOG.warning("seed_num is None. Setting it to 0.")
        random_state = 0
    search_space = {
        "random_state": random_state,
        "lr": .01,
        "momentum": ray.tune.uniform(0.1, 0.9),
        "wd": ray.tune.loguniform(1e-6, 1e-2),
        "warm_start_epochs": ray.tune.choice([10, 20, 50]),
        "epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
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

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_l_auc", "val_l_accuracy", "val_l_APS"])
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    execution_start_time = time.time() 
    result = ray.tune.run(
        LPU.scripts.mpe.run_mpe.train_model,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        scheduler=scheduler,
        mode='min',
        local_dir=results_dir,
        progress_reporter=reporter,
)
    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")
    
    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_trial_report = {
        "Best trial config": best_trial.config,
        "Best trial final validation loss": best_trial.last_result["val_overall_loss"],
        "Best trial final test scores": {
            "test_overall_loss": best_trial.last_result["test_overall_loss"],
            "test_y_auc": best_trial.last_result["test_y_auc"],
            "test_y_accuracy": best_trial.last_result["test_y_accuracy"],
            "test_y_APS": best_trial.last_result["test_y_APS"]
        },
        "Execution time": execution_time,
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
