import sys
import json
import ray.tune
import ray.train

import LPU.scripts.uPU.run_uPU

def main(num_samples=50, max_num_epochs=100, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "learning_rate": ray.tune.loguniform(1e-4, 1e-1),
        "epoch": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "gamma": ray.tune.uniform(0.1, 1.0),
        "beta": ray.tune.uniform(0.0, 1.0),
        "loss": ray.tune.choice(['sigmoid']),
        "batch_size": {
            "train": ray.tune.choice([32, 64, 128]),
            "test": ray.tune.choice([32, 64, 128]),
            "val": ray.tune.choice([32, 64, 128]),
            "holdout": ray.tune.choice([32, 64, 128])
        },
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS",
        "test_overall_loss", "test_y_auc", "test_y_accuracy", "test_y_APS"])

    result = ray.tune.run(
        LPU.scripts.uPU.run_uPU.train_model,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        mode='min',
        local_dir=results_dir,
        progress_reporter=reporter,
        max_concurrent_trials=3)

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
    print (json.dumps(best_trial_report, indent=4))
    return best_trial_report

if __name__ == "__main__":
    main()
