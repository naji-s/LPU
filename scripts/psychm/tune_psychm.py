import json
import ray.tune
import ray.train

import LPU.scripts.psychm.run_psychm

def main(num_samples=1, max_num_epochs=10, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "inducing_points_size": ray.tune.choice([32, 64]),
        "learning_rate": ray.tune.loguniform(1e-4, 1e-1),
        "num_epochs": ray.tune.choice(range(5, max_num_epochs)),
        "intrinsic_kernel_params": {
            "normed": ray.tune.choice([True, False]),
            "noise_factor": ray.tune.uniform(0, 0.1),
            "amplitude": ray.tune.uniform(0.1, 1),
            "n_neighbor": ray.tune.randint(3, 10),
            "lengthscale": ray.tune.loguniform(1e-2, 1),
            "neighbor_mode": ray.tune.choice(['connectivity', 'distance']),
            "power_factor": ray.tune.randint(1, 3),
            "invert_M_first": ray.tune.choice([True, False]),
            "normalize": ray.tune.choice([True, False])
        },
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
        LPU.scripts.psychm.run_psychm.train_model,
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
            "test_y_APS": best_trial.last_result["test_y_APS"]
        }
    }
    print (json.dumps(best_trial_report, indent=4))
    return best_trial_report

if __name__ == "__main__":
    main()
