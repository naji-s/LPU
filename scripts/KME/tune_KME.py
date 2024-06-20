import json
import os
import datetime
import time

import torch
import ray.train
import ray.tune
import ray.tune.schedulers

import LPU.models
import LPU.models.geometric
import LPU.models.geometric.KME
import LPU.models.geometric.KME.KME
import LPU.scripts
import LPU.scripts.KME.run_KME
import LPU.utils.dataset_utils
import LPU.utils.utils_general


LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'KME'


def reload_and_validate_model(results_dir, dataloaders_dict):
    # Load the best trial report
    best_trial_report_path = os.path.join(results_dir, MODEL_NAME)
    best_trial_report_file = [os.path.join(best_trial_report_path, f, 'best_trial_results.json') for f in os.listdir(best_trial_report_path)][0]

    with open(best_trial_report_file, "r") as json_file:
        best_trial_report = json.load(json_file)

    best_trial_config = best_trial_report["Best trial config"]

    # Load the checkpoint
    checkpoint_dir = os.path.join(results_dir, best_trial_config["checkpoint_path"], "checkpoint.pt")
    checkpoint = torch.load(checkpoint_dir)

    # Recreate the model
    model = LPU.models.geometric.KME.KME.KME()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # Set the model to evaluation mode

    # Validate the model
    test_dataloader = dataloaders_dict['test']
    test_results = LPU.scripts.KME.run_KME.validate_model(model, test_dataloader)

    print(json.dumps(test_results, indent=4))
    return test_results

def main(num_samples=100, max_num_epochs=200, gpus_per_trial=0, results_dir=None, random_state=None):
    # Configuration for hyperparameters to be tuned
    if random_state is None:
        LOG.warning("seed_num is None. Setting it to 0.")
        random_state = 0
    search_space = {
        "input_dim": 4096,
        "random_state": random_state,
        "inducing_points_size": ray.tune.choice([64]),
        "learning_rate": .01,
        "num_epochs": ray.tune.choice(range(max_num_epochs, max_num_epochs + 1)),
        "kernel_mode": ray.tune.choice([2]),
        "intrinsic_kernel_params": {
            "normed": ray.tune.choice([False]),
            "kernel_type": ray.tune.choice(["laplacian"]),
            "noise_factor": ray.tune.uniform(1e-4, 1e-1),
            "amplitude": ray.tune.loguniform(0.01, 100.0),
            "n_neighbor": ray.tune.choice([5]),
            "lengthscale": ray.tune.loguniform(0.01, 100.0),
            "neighbor_mode": ray.tune.choice(["distance", "connectivity"]),
            "power_factor": ray.tune.uniform(0.1, 2.0),
        },
        "lambda_0": ray.tune.loguniform(1 + 1e-4, 1 + 1e-1),
        "lambda_1_increment": ray.tune.loguniform(1e-4, 1 + 1e-1),
        "thres_par": ray.tune.uniform(0.1, 1.0),
    }
    data_config =  {       
                    "dataset_name": "animal_no_animal",  # fashionMNIST
                    "dataset_kind": "LPU",
                    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
                    "device": "cpu",
                    'ratios': 
                    {
                        # *** NOTE ***
                        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
                        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
                        'test': 0.4,
                        'val': 0.05,
                        'holdout': .05,
                        'train': .5, 
                    },
                    "batch_size": {
                        "train": 64,
                        "test": 64,
                        "val": 64,
                        "holdout": 64
                    }        
                }
    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_y_auc", "val_y_accuracy", "val_y_APS"])
    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    execution_start_time = time.time()
    dataloaders_dict = LPU.utils.dataset_utils.create_dataloaders_dict(data_config)
    result = ray.tune.run(
        ray.tune.with_parameters(LPU.scripts.KME.run_KME.train_model, dataloaders_dict=dataloaders_dict, with_ray=True),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=search_space,
        num_samples=num_samples,
        metric='val_overall_loss',
        scheduler=scheduler,
        mode='min',
        storage_path=results_dir,
        progress_reporter=reporter,
        keep_checkpoints_num=1
    )
    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")

    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_model_state = torch.load(os.path.join(best_trial.checkpoint.path, "checkpoint.pt"))
    breakpoint()
    
    best_model = LPU.models.geometric.KME.KME.KME(config=best_trial.config)
    all_config = search_space.copy()
    all_config.update(data_config)

    best_model.load_state_dict(best_model_state["model_state"])

    best_model_test_results = best_model.validate(dataloaders_dict['test'], model=best_model.gp_model, loss_fn=best_model.loss_fn)
    final_epoch = best_trial.last_result["training_iteration"]
    for key in best_trial.last_result:
        if 'val_' in key:
            best_trial.last_result[key.replace('val', 'test')] = best_model_test_results['test'][''.join(key.split('_')[1:])]
    
    best_trial_report = {
        "Best trial config": best_trial.config,
        "Best trial final validation loss": best_trial.last_result["val_overall_loss"],
        "Best trial final test scores": {
            "test_overall_loss": best_trial.last_result["test_overall_loss"],
            "test_y_auc": best_trial.last_result["test_y_auc"],
            "test_y_accuracy": best_trial.last_result["test_y_accuracy"],
            "test_y_APS": best_trial.last_result["test_y_APS"]
        },
        "Execution Time": execution_time,
        "Final epoch": final_epoch,
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
