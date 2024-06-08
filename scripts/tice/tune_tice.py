import json
import os
import datetime

import ray.tune
import ray.train

import LPU.scripts
import LPU.scripts.tice
import LPU.scripts.tice.run_tice
import LPU.utils.utils_general

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'tice'

def main(num_samples=50, max_num_epochs=100, gpus_per_trial=0, results_dir=None, random_state=None):
    # Configuration for hyperparameters to be tuned
    if random_state is None:
        LOG.warning("seed_num is None. Setting it to 0.")
        random_state = 0
    search_space = {
        "random_state": random_state,
        "inducing_points_size": ray.tune.choice([16, 32, 64]),
        "learning_rate": ray.tune.loguniform(1e-4, 1e-1),
        "num_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "batch_size": {
            "train": ray.tune.choice([32, 64, 128]),
            "test": ray.tune.choice([32, 64, 128]),
            "val": ray.tune.choice([32, 64, 128]),
            "holdout": ray.tune.choice([32, 64, 128])
        },
        "delta": ray.tune.choice([None, 0.1, 0.01]),
        "max-bepp": ray.tune.randint(1, 10),
        "maxSplits": ray.tune.randint(100, 1000),
        "minT": ray.tune.randint(5, 20),
        "nbIts": ray.tune.randint(1, 5),
        "intrinsic_kernel_params": {
            "normed": ray.tune.choice([True, False]),
            "kernel_type": ray.tune.choice(["laplacian"]),
            # "heat_temp": ray.tune.loguniform(1e-3, 1e-1),
            "noise_factor": ray.tune.uniform(0, 0.1),
            "amplitude": ray.tune.uniform(0.1, 1),
            "n_neighbor": ray.tune.randint(3, 10),
            "lengthscale": ray.tune.loguniform(1e-2, 1),
            "neighbor_mode": ray.tune.choice(['connectivity', 'distance']),
            "power_factor": ray.tune.randint(1, 3),
            "invert_M_first": ray.tune.choice([True, False]),
            "normalize": ray.tune.choice([True, False])        
        }
    }
    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS",
        "test_overall_loss", "test_y_auc", "test_y_accuracy", "test_y_APS"])

    result = ray.tune.run(
        LPU.scripts.tice.run_tice.train_model,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        mode='min',
        local_dir=results_dir,
        progress_reporter=reporter,
        max_concurrent_trials=8)

    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_trial_report = {
        "Best trial config": best_trial.config,
        "Best trial final validation loss": best_trial.last_result["val_overall_loss"],
        "Best trial final test scores": {
            "test_overall_loss": best_trial.last_result["test_overall_loss"],
            "test_y_auc": best_trial.last_result["test_y_auc"],
            "test_y_accuracy": best_trial.last_result["test_y_accuracy"],
            "test_y_APS": best_trial.last_result["test_y_APS"]}
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
