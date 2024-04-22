import torch

import lpu.constants
import lpu.datasets.LPUDataset
import lpu.models.geometric.psychmGGPC
import lpu.utils.utils_general
import lpu.datasets.dataset_utils

LEARNING_RATE = 0.01
BATCH_SIZE = 32
DEVICE = 'cpu'
INDUCING_POINTS_SIZE = 32
torch.set_default_dtype(lpu.constants.DTYPE)
def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/psychm_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    inducing_points_size = config.get('inducing_points_size', INDUCING_POINTS_SIZE)
    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False)
    train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE, hold_out_size=ELKAN_HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    inducing_points_initial_vals = lpu.datasets.dataset_utils.initialize_inducing_points(train_loader, inducing_points_size)
    psychmGVGP_model = lpu.models.geometric.psychmGGPC.PsychMGP(config, inducing_points_initial_vals=inducing_points_initial_vals, training_size=len(train_loader.dataset), num_features=inducing_points_initial_vals.shape[-1])

    learning_rate = config.get('learning_rate', LEARNING_RATE)
    optimizer = torch.optim.Adam([{
        'params': psychmGVGP_model.parameters(), 'lr': learning_rate
    }])
    for i in range(50):
        psychmGVGP_model.train_one_epoch(optimizer=optimizer, dataloader=train_loader)
        psychmGVGP_model.validate(val_loader)

if __name__ == "__main__":
    main()

