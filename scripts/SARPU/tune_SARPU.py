import json
import os
import datetime
import time

import ray.train
import ray.tune
import ray.tune.schedulers

import LPU.scripts
import LPU.scripts.sarpu
import LPU.scripts.sarpu.run_sarpu_em
import LPU.utils.utils_general

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'sarpu'

def main(num_samples=100, max_num_epochs=200, gpus_per_trial=0, results_dir=None, random_state=None):
    # setting the seed for the tuning
    if random_state is not None:
        LPU.utils.utils_general.set_seed(random_state)


    # Configuration for hyperparameters to be tuned
    search_space = {
        # making sure the model training is not gonna set the seed 
        # since we potentially might want to set the seed for the tuning
		"random_state": None,
        "num_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "SAR_PU_classification_model": ray.tune.choice(['logistic']),
        "svm_params": {
            "tol": ray.tune.loguniform(1e-5, 1e-3),
            "C": ray.tune.loguniform(0.1, 10),
            "kernel": ray.tune.choice(['linear', 'poly', 'rbf', 'sigmoid']),
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
            "max_iter": ray.tune.choice([-1, 1000, 2000, 5000]),
        },
        "batch_size": {
            "train": ray.tune.choice([64]),
            "test": ray.tune.choice([64]),
            "val": ray.tune.choice([64]),
            "holdout": ray.tune.choice([64])
        },
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS",
        "test_overall_loss", "test_y_auc", "test_y_accuracy", "test_y_APS"])
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    execution_start_time = time.time()
    result = ray.tune.run(
        LPU.scripts.sarpu.run_sarpu_em.train_model,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        scheduler=scheduler,
        mode='min',
        storage_path=results_dir,
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
        "test_y_APS": best_trial.last_result["test_y_APS"]},
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
