import logging
import os
import tempfile

import ray.train

import torch

import lpu.constants
import lpu.models.psychm.psychm
import lpu.models.psychm.run_psychm
import lpu.utils.utils_general

# Create a logger instance
LOG = logging.getLogger(__name__)

LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
BATCH_SIZE = 32
TRAIN_VAL_RATIO = .1
ELKAN_HOLD_OUT_SIZE = 0.
TRAIN_TEST_RATIO = .5

INTRINSIC_KERNEL_PARAMS = {
    'normed': False,
    'kernel_type': 'laplacian',
    'heat_temp': .01,
    'noise_factor': 0.,
    'amplitude':  0.5,
    'n_neighbor': 10,
    'lengthscale':  0.3, 
    'neighbor_mode': 'distance',
    'power_factor': 1,
    'invert_M_first': False 
}

def tune_train_psychm(config={}):
    # Instantiate your SKLearnCompatibleGP with the provided configuration
    model = lpu.models.psychm.psychm.PsychMGP(dtype=config['dtype'], train_val_ratio=config['train_val_ratio'], 
                     device=config['device'], learning_rate=config['learning_rate'], 
                     num_epochs=config['num_epochs'], evaluation_interval=config['evaluation_interval'], 
                     epoch_blocks=config['epoch_blocks'], intrinsic_kernel_params=config['intrinsic_kernel_params'])
    model._setup(config=config)    
    # Fit the model
    epoch = 1
    while True:        
        model._train_one_epoch()
        val_loss, val_auc = model._validate()  # Assuming `score` returns an evaluation metric
    
        # In Ray Tune, you report results with `tune.report`
        # ray.train.report({"val_loss":val_loss, "training_iteration": epoch, 'val_auc': val_auc})
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if (epoch + 1) % 5 == 0:
                # This saves the model to the trial directory
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                torch.save(
                    model.gp_model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "gp_model.pth")
                )
                torch.save(
                    model.likelihood.state_dict(),
                    os.path.join(temp_checkpoint_dir, "likelihood.pth")
                )

                # Send the current training result back to Tune
            ray.train.report({"val_loss":val_loss.item(), "training_iteration": epoch, 'val_auc': val_auc}, checkpoint=checkpoint)        
        epoch += 1


if __name__ == "__main__":
        # Define your configuration
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/psychm_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)

    # tune_train_psychm(config=config)
    # Then, use this function with Ray Tune
    ray.tune.run(
    tune_train_psychm,
    config=config,  # Your configuration here,
    stop={"training_iteration": 10},
    )
