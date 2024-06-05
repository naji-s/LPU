import json
import sys
sys.path.append('lpu/external_libs/PU_learning')
sys.path.append('lpu/external_libs/PU_learning/data_helper')

import ray.tune
import ray.train

import lpu.scripts.mpe.run_mpe

def main(num_samples=1, max_num_epochs=10, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "lr": ray.tune.loguniform(1e-4, 1e-1),
        "momentum": ray.tune.uniform(0.1, 0.9),
        "wd": ray.tune.loguniform(1e-6, 1e-2),
        "warm_start_epochs": ray.tune.choice([5, 10, 20]),
        "epochs": ray.tune.choice(range(5, max_num_epochs)),
        "alpha": ray.tune.uniform(0.3, 0.7),
        "beta": ray.tune.uniform(0.3, 0.7),
        "train_method": ray.tune.choice(["TEDn", "alternative"]),
        "net_type": ray.tune.choice(["LeNet", "CustomNet"]),
        "sigmoid_loss": ray.tune.choice([True, False]),
        "estimate_alpha": ray.tune.choice([True, False]),
        "warm_start": ray.tune.choice([True, False]),
        "batch_size": {
            "train": ray.tune.choice([32, 64, 128]),
            "test": ray.tune.choice([32, 64, 128]),
            "val": ray.tune.choice([32, 64, 128]),
            "holdout": ray.tune.choice([32, 64, 128])
        },
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_l_auc", "val_l_accuracy", "val_l_APS"])

    result = ray.tune.run(
        lpu.scripts.mpe.run_mpe.train_model,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        local_dir=results_dir,
        progress_reporter=reporter,
        max_concurrent_trials=5)

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
