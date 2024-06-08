import json
import ray.tune
import ray.tune.schedulers
import ray.train
import numpy as np

import LPU.scripts.psychm.run_psychm

def main(num_samples=50, max_num_epochs=100, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "inducing_points_size": ray.tune.choice([64]),
        "learning_rate": ray.tune.loguniform(1e-3 * 2, 1e-2 * 2),
        "num_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "intrinsic_kernel_params": {
            "normed": ray.tune.choice([True, False]),
            "noise_factor": ray.tune.uniform(0, 0.1),
            "amplitude": ray.tune.uniform(0.1, 1),
            "n_neighbor": ray.tune.randint(5, 10),
            "lengthscale": ray.tune.loguniform(1e-2, 1), 
            "neighbor_mode": ray.tune.choice(['connectivity', 'distance']),
            "power_factor": ray.tune.randint(1, 3),
        },
        # "batch_size": {
        #     "train": ray.tune.choice([32, 64, 128]),
        #     "test": ray.tune.choice([32, 64, 128]),
        #     "val": ray.tune.choice([32, 64, 128]),
        #     "holdout": ray.tune.choice([32, 64, 128])
        # },
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS",
        "test_overall_loss", "test_y_auc", "test_y_accuracy", "test_y_APS"])

    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    try:
        result = ray.tune.run(
            LPU.scripts.psychm.run_psychm.train_model,
            resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
            config=search_space,
            num_samples=num_samples,
            metric='val_overall_loss',
            mode='min',
            scheduler=scheduler,
            fail_fast=False,
            local_dir=results_dir,
            progress_reporter=reporter)

        # Filter and find the best trial that completed successfully
        successful_trials = [trial for trial in result.trials if trial.status == "TERMINATED"]
        best_trial = min(successful_trials, key=lambda trial: trial.last_result["val_overall_loss"]) if successful_trials else None

        if best_trial:
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
        else:
            print("No successful trials were found.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()