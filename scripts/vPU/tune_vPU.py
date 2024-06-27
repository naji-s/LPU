import json
import ray.train
import ray.tune
import ray.tune.schedulers
import os
import datetime
import time
import tempfile
import torch

from matplotlib import pyplot as plt
import LPU.scripts
import LPU.scripts.vPU
import LPU.scripts.vPU.run_vPU
import LPU.models.vPU.vPU
import LPU.utils.dataset_utils
import LPU.utils.utils_general
import LPU.models.MPE.MPE
import LPU.utils.ray_utils

LOG = LPU.utils.utils_general.configure_logger(__name__)
MODEL_NAME = 'vPU'

def SET_DEFAULT_SEARCH_SPACE():
    default_search_space = {
        "random_state": None,
        "learning_rate": .01,
        "epochs": ray.tune.choice(range(100, 100 + 1)),
        "mix_alpha": ray.tune.uniform(0.1, 0.5),
        "lam": ray.tune.uniform(0.1, 0.9),
    }
    return default_search_space

def main(config=None, num_samples=100, max_num_epochs=200, gpus_per_trial=0, results_dir=None, random_state=None, tune=False):
    # setting the seed for the tuning
    if random_state is not None:
        LPU.utils.utils_general.set_seed(random_state)

    DEFAULT_SEARCH_SPACE = SET_DEFAULT_SEARCH_SPACE()
    if config is None:
        config = {}
    if tune:
        config = LPU.utils.utils_general.deep_update(DEFAULT_SEARCH_SPACE, config)
    else:
        config = LPU.utils.utils_general.deep_update(LPU.models.vPU.vPU.DEFAULT_CONFIG, config)

    data_config = {
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
        },
        "drop_last":
            {
                "train": True,
                "test": False,
                "val": False,
                "holdout": False
            }
    }
    experiment_name = f'train_{MODEL_NAME}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    EXPERIMENT_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json_save_dir = os.path.join(results_dir, MODEL_NAME, EXPERIMENT_DATETIME)
    # Ensure the directory exists
    os.makedirs(json_save_dir, exist_ok=True)

    reporter = ray.tune.CLIReporter(metric_columns=[
        "val_overall_loss", "val_phi_loss", "val_reg_loss",
        "test_overall_loss", "test_phi_loss", "test_reg_loss"])

    scheduler = ray.tune.schedulers.ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)

    execution_start_time = time.time()
    positive_label_list = LPU.scripts.vPU.run_vPU.get_positive_labels(data_config['dataset_name'])
    if data_config['dataset_kind'] in ['LPU', 'MPE']:
        dataloaders_dict = LPU.models.MPE.MPE.create_dataloaders_dict_mpe(data_config)
    else:
        get_loaders = LPU.scripts.vPU.run_vPU.get_loaders_by_dataset_name(data_config['dataset_name'])
        # TODO: make sure the datasets are balanced coming out of this
        x_loader, p_loader, val_x_loader, val_p_loader,  test_loader, idx = get_loaders(batch_size=data_config['batch_size']['train'], num_labeled=data_config['num_labeled'], positive_label_list=positive_label_list)
        dataloaders_dict = {}
        dataloaders_dict['train'] = {}
        dataloaders_dict['train']['PDataset'] = p_loader
        dataloaders_dict['train']['UDataset'] = x_loader
        dataloaders_dict['val'] = {}
        dataloaders_dict['val']['PDataset'] = val_p_loader
        dataloaders_dict['val']['UDataset'] = val_x_loader   
    result = ray.tune.run(
        ray.tune.with_parameters(LPU.scripts.vPU.run_vPU.train_model, dataloaders_dict=dataloaders_dict, with_ray=True),
    resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    metric='val_overall_loss',
    scheduler=scheduler,
    mode='min',
    storage_path=results_dir,
    progress_reporter=reporter,
    keep_checkpoints_num=1,
    name=experiment_name,
    callbacks=[LPU.utils.ray_utils.PlotMetricsCallback(allow_list=['time_this_iter_s', 'epoch', 'val_overall_loss', 'val_phi_loss', 'val_reg_loss', 'val_y_auc', 'val_y_accuracy', 'val_y_APS'], json_save_dir=json_save_dir)],
    )
    execution_time = time.time() - execution_start_time
    LOG.info(f"Execution time: {execution_time} seconds")
    best_trial = result.get_best_trial("val_overall_loss", "min", "last")
    best_model_checkpoint = torch.load(os.path.join(best_trial.checkpoint.path, "checkpoint.pt"))
    best_model = LPU.models.vPU.vPU.vPU(config=best_model_checkpoint["config"], 
                                        input_dim=dataloaders_dict['train']['UDataset'].dataset.data.shape[1])
    best_model.load_state_dict(best_model_checkpoint["model_state"])

    best_model_test_results = best_model.validate(train_p_loader=dataloaders_dict['train']['PDataset'],
                                                train_u_loader=dataloaders_dict['train']['UDataset'],
                                                val_p_loader=dataloaders_dict['test']['PDataset'],
                                                val_u_loader=dataloaders_dict['test']['UDataset'], epoch=None, train_phi_loss=None,
                                                var_loss=None, train_reg_loss=None, test_loader=None)
    final_epoch = best_trial.last_result["training_iteration"]
    final_results = best_trial.last_result.copy()
    for key in best_trial.last_result:
        if 'val_' in key:
            final_results[key.replace('val', 'test')] = best_model_test_results['_'.join(key.split('_')[1:])]
    best_trial_report = {
    "Best trial config": best_trial.config,
    "Best trial final validation loss": final_results["val_overall_loss"],
    "Best trial final test scores": {
        "test_overall_loss": final_results["test_overall_loss"],        
        "test_y_auc": final_results["test_y_auc"],
        "test_y_accuracy": final_results["test_y_accuracy"],
        "test_y_APS": final_results["test_y_APS"]
        },
        # "test_phi_loss": final_results["test_phi_loss"],
        # "test_reg_loss": final_results["test_reg_loss"]},
    "Execution Time": execution_time,
    "Final epoch": final_epoch
    }    

    print(json.dumps(best_trial_report, indent=4, cls=LPU.utils.utils_general.CustomJSONEncoder))
    return best_trial_report

if __name__ == "__main__":
    args = LPU.utils.utils_general.tune_parse_args()
    main(num_samples=args.num_samples, max_num_epochs=args.max_num_epochs,
         gpus_per_trial=args.gpus_per_trial, results_dir=args.results_dir,
         random_state=args.random_state, tune=args.tune)
