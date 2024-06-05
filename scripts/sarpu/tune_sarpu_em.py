import sys
import ray.tune
import ray.train

import LPU.scripts
import LPU.scripts.sarpu
import LPU.scripts.sarpu.run_sarpu_em

def main(num_samples=1, max_num_epochs=10, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "num_epochs": ray.tune.choice(range(5, max_num_epochs)),
        "SAR_PU_classification_model": ray.tune.choice(['logistic']),
        "svm_params": {
            "tol": ray.tune.loguniform(1e-5, 1e-3),
            "C": ray.tune.loguniform(0.1, 10),
            "kernel": ray.tune.choice(['linear', 'poly', 'rbf', 'sigmoid']),
            "degree": ray.tune.randint(2, 5),
            "gamma": ray.tune.loguniform(1e-4, 1),
            "max_iter": ray.tune.choice([-1, 1000, 2000, 5000]),
        },
        "logistic_params": {
            "penalty": ray.tune.choice(['l2']),
            "tol": ray.tune.loguniform(1e-5, 1e-3),
            "C": ray.tune.loguniform(0.1, 10),
            "fit_intercept": ray.tune.choice([True, False]),
            "solver": ray.tune.choice(['lbfgs']),
            "max_iter": ray.tune.choice(range(5, max_num_epochs)),
        },
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS",
        "test_overall_loss", "test_y_auc", "test_y_accuracy", "test_y_APS"])

    result = ray.tune.run(
        LPU.scripts.sarpu.run_sarpu_em.train_model,
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
    return best_trial_report

if __name__ == "__main__":
    main()