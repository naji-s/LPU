import sys
import ray.tune
import ray.train

import lpu.scripts
import lpu.scripts.vpu
import lpu.scripts.vpu.run_vpu

def main(num_samples=1, max_num_epochs=10, gpus_per_trial=0, results_dir=None):
    # Configuration for hyperparameters to be tuned
    search_space = {
        "learning_rate": ray.tune.loguniform(1e-5, 1e-3),
        "epochs": ray.tune.choice(range(5, max_num_epochs)),
        "batch_size": {
            "train": ray.tune.choice([32, 64, 128]),
        },
        "num_labeled": ray.tune.choice([100, 200, 500]),
        "mix_alpha": ray.tune.uniform(0.1, 0.5),
        "lam": ray.tune.uniform(0.1, 0.9),
    }

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_phi_loss", "val_reg_loss",
        "test_overall_loss", "test_phi_loss", "test_reg_loss"])

    result = ray.tune.run(
        lpu.scripts.vpu.run_vpu.train_model,
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