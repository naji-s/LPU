import sys
import json
import yaml
import ray.tune
import ray.train

import lpu.scripts
import lpu.scripts.distPU
import lpu.scripts.distPU.run_distPU

def main(num_samples=1, max_num_epochs=10, gpus_per_trial=0, results_dir=None):
    search_space = {
        "warm_up_lr": ray.tune.loguniform(1e-4, 1e-1),
        "lr": ray.tune.loguniform(1e-4, 1e-1),
        "warm_up_weight_decay": ray.tune.loguniform(1e-6, 1e-2),
        "weight_decay": ray.tune.loguniform(1e-6, 1e-2),
        "warm_up_epochs": ray.tune.choice(range(5, max_num_epochs, 5)),
        "pu_epochs": ray.tune.choice(range(5, max_num_epochs, 5)),
        "co_entropy": ray.tune.uniform(0.001, 0.01),
        "num_labeled": ray.tune.choice([1000, 5000, 10000]),
        "optimizer": ray.tune.choice(['adam', 'sgd']),
        "schedular": ray.tune.choice(['cos-ann', 'step']),
        "entropy": ray.tune.uniform(0.1, 1.0),
        "co_mu": ray.tune.uniform(0.001, 0.01),
        "alpha": ray.tune.uniform(3.0, 9.0),
        "co_mix_entropy": ray.tune.uniform(0.01, 0.1),
        "co_mixup": ray.tune.uniform(2.0, 10.0),
        "ratios": {
            "test": ray.tune.uniform(0.1, 0.3),
            "val": ray.tune.uniform(0.1, 0.3),
            "holdout": ray.tune.uniform(0.05, 0.1),
            "train": ray.tune.uniform(0.3, 0.5)
        },
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS"])

    result = ray.tune.run(
        lpu.scripts.distPU.run_distPU.train_model,
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
