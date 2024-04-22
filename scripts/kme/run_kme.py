import logging

import lpu.models.geometric.KMEGGPC
LOG = logging.getLogger(__name__)

import torch
# import torch.optim

import lpu.constants
import lpu.datasets.LPUDataset
import lpu.datasets.dataset_utils
import lpu.models.geometric.elkanGGPC
import lpu.utils.utils_general

LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
BATCH_SIZE = 32
TRAIN_VAL_RATIO = .1
ELKAN_HOLD_OUT_SIZE = 0.1
TRAIN_TEST_RATIO = .5

def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/psychm_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    config['kernel_mode'] = 2
    inducing_points_size = config.get('inducing_points_size', INDUCING_POINTS_SIZE)
    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False)
    train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE, hold_out_size=ELKAN_HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    inducing_points_initial_vals = lpu.datasets.dataset_utils.initialize_inducing_points(train_loader, inducing_points_size)
    elkan_model = lpu.models.geometric.KMEGGPC.KMEModelGGPC(config, inducing_points_initial_vals=inducing_points_initial_vals, training_size=len(train_loader.dataset))
    learning_rate = config.get('learning_rate', LEARNING_RATE)
    optimizer = torch.optim.Adam([{
        'params': elkan_model.parameters(), 'lr': learning_rate
    }])
    for i in range(50):
        elkan_model.train_one_epoch(optimizer=optimizer, dataloader=train_loader)
        elkan_model.set_C(holdout_loader)
        print (elkan_model.validate(val_loader))

if __name__ == "__main__":
    main()

