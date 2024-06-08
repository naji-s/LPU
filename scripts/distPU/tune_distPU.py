import json
import os
import datetime

import ray.tune
import ray.train

import LPU.scripts
import LPU.scripts.distPU
import LPU.scripts.distPU.run_distPU
import LPU.utils.utils_general


LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'distPU'

def main(num_samples=50, max_num_epochs=100, gpus_per_trial=0, results_dir=None, random_state=None):
    if random_state is None:
        LOG.warning("seed_num is None. Setting it to 0.")
        random_state = 0
    search_space = {
        "random_state": random_state,
        "warm_up_lr": ray.tune.loguniform(1e-4, 1e-1),
        "lr": ray.tune.loguniform(1e-3, 1e-1),
        # "warm_up_weight_decay": ray.tune.loguniform(1e-6, 1e-2),
        # "weight_decay": ray.tune.loguniform(1e-6, 1e-2),        
        "pu_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "warm_up_epochs": ray.tune.choice([10, 20, 50]),
        "co_entropy": ray.tune.uniform(0.001, 0.01),
        "num_labeled": ray.tune.choice([1000, 5000, 10000]),
        "optimizer": ray.tune.choice(['adam']),
        # "schedular": ray.tune.choice(['cos-ann', 'step']),
        "entropy": ray.tune.uniform(0.1, 1.0),
        "co_mu": ray.tune.uniform(0.001, 0.01),
        "co_mix_entropy": ray.tune.uniform(0.01, 0.1),
        "co_mixup": ray.tune.uniform(2.0, 10.0),
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS"])

    result = ray.tune.run(
        LPU.scripts.distPU.run_distPU.train_model,
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
