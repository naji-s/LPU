import json
import ray.tune
import ray.train

import LPU.scripts
import LPU.scripts.kme.run_kme

def main(num_samples=50, max_num_epochs=100, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "inducing_points_size": ray.tune.choice([16, 32, 64]),
        "learning_rate": ray.tune.loguniform(1e-4, 1e-1),
        "num_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "kernel_mode": ray.tune.choice([1, 2, 3]),
        "intrinsic_kernel_params": {
            "normed": ray.tune.choice([True, False]),
            "heat_temp": ray.tune.uniform(0.001, 0.1),
            "noise_factor": ray.tune.uniform(0.0, 0.1),
            "amplitude": ray.tune.uniform(0.1, 1.0),
            "n_neighbor": ray.tune.choice([3, 5, 10]),
            "lengthscale": ray.tune.uniform(0.1, 1.0),
            "neighbor_mode": ray.tune.choice(["distance", "connectivity"]),
            "power_factor": ray.tune.uniform(0.1, 2.0),
        },
        "batch_size": {
            "train": ray.tune.choice([32, 64, 128]),
            "test": ray.tune.choice([32, 64, 128]),
            "val": ray.tune.choice([32, 64, 128]),
            "holdout": ray.tune.choice([32, 64, 128])
        },
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS"])

    result = ray.tune.run(
        LPU.scripts.kme.run_kme.train_model,
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
    print(json.dumps(best_trial_report, indent=4))
    return best_trial_report

if __name__ == "__main__":
    main()
