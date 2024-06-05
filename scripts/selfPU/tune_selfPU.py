import json
import ray.tune
import ray.train

import LPU.scripts
import LPU.scripts.selfPU
import LPU.scripts.selfPU.run_selfPU

def main(num_samples=1, max_num_epochs=10, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "lr": ray.tune.loguniform(1e-4, 1e-1),
        "weight_decay": ray.tune.loguniform(1e-6, 1e-3),
        "epochs": ray.tune.randint(5, max_num_epochs),
        "weight": ray.tune.uniform(0.5, 2.0),
        "self_paced_start": ray.tune.randint(5, 20),
        "self_paced_stop": ray.tune.randint(5, max_num_epochs),
        "self_paced_frequency": ray.tune.randint(5, 15),
        "self_paced_type": ray.tune.choice(["A", "B", "C"]),
        "ema_decay": ray.tune.loguniform(0.99, 0.999),
        "consistency": ray.tune.uniform(0.1, 0.5),
        "consistency_rampup": ray.tune.randint(200, 600),
        "top1": ray.tune.uniform(0.2, 0.6),
        "top2": ray.tune.uniform(0.4, 0.8),
        "alpha": ray.tune.loguniform(1e-2, 1.0),
        "gamma": ray.tune.loguniform(1e-3, 0.1),
        "num_p": ray.tune.randint(500, 2000),
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS",
        "test_overall_loss", "test_y_auc", "test_y_accuracy", "test_y_APS"])

    result = ray.tune.run(
        LPU.scripts.selfPU.run_selfPU.train_model,
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
    print (json.dumps(best_trial_report, indent=4))
    return best_trial_report

if __name__ == "__main__":
    main()