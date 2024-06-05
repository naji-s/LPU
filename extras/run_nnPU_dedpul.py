import logging
import sys

sys.path.append('LPU/external_libs/DEDPUL')

import LPU.models.geometric.KMEGGPC
LOG = logging.getLogger(__name__)

# import torch.optim

import LPU.constants
import LPU.datasets.LPUDataset
import LPU.utils.dataset_utils
import LPU.models.nnPU_dedpul
import LPU.utils.utils_general

LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
TRAIN_VAL_RATIO = .1
HOLD_OUT_SIZE = None
TRAIN_TEST_RATIO = .5

def main():
    yaml_file_path = '/Users/naji/phd_codebase/LPU/configs/nnPU_config.yaml'
    config = LPU.utils.utils_general.load_and_process_config(yaml_file_path)
    config['kernel_mode'] = 2
    lpu_dataset = LPU.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False, invert_l=False)
    BATCH_SIZE = len(lpu_dataset)
    train_loader, test_loader, val_loader, holdout_loader = LPU.utils.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE, hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    nnPU_model = LPU.models.nnPU_dedpul.nnPU(config)
    preds = nnPU_model.train(train_loader)
    nnPU_model.set_C(train_loader, preds)
    print(nnPU_model.validate(val_loader))
    
if __name__ == "__main__":
    main()

