import argparse
import datetime
import json
import os

import ray.tune
import ray.train

import LPU.scripts
import LPU.scripts.dedpul
import LPU.scripts.dedpul.run_dedpul
import LPU.constants
import LPU.utils
import LPU.utils.utils_general

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'dedpul'


def main(num_samples=50, max_num_epochs=100, gpus_per_trial=0, results_dir=None, random_state=None):
    if random_state is None:
        LOG.warning("seed_num is None. Setting it to 0.")
        random_state = 0
    search_space = {
        "random_state": random_state,
        "learning_rate": ray.tune.loguniform(1e-4, 1e-1),
        # "batch_size": {
        #     "train": ray.tune.choice([16, 32, 64]),
        # },
        "num_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "nrep": ray.tune.choice([5, 10, 20]),
        "estimate_diff_options": {
            "MT": ray.tune.choice([True, False]),
            "MT_coef": ray.tune.uniform(0.1, 0.5),
            "decay_MT_coef": ray.tune.choice([True, False]),
            "bw_mix": ray.tune.uniform(0.01, 0.1),
            "bw_pos": ray.tune.uniform(0.05, 0.2),
            "n_gauss_mix": ray.tune.choice([10, 20, 30]),
            "n_gauss_pos": ray.tune.choice([5, 10, 15]),
            "bins_mix": ray.tune.choice([10, 20, 30]),
            "bins_pos": ray.tune.choice([10, 20, 30]),
            "k_neighbours": ray.tune.choice([5, 10])
        },
        "base_config_file_path": "/Users/naji/phd_codebase/LPU/configs/dedpul_config.yaml",
        "random_state": random_state
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS"])

    result = ray.tune.run(
        LPU.scripts.dedpul.run_dedpul.train_model,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
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
            "test_y_APS": best_trial.last_result["test_y_APS"]
        }
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
