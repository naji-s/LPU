import copy
import logging
import unittest.mock
import types 

import numpy as np
import torch.optim

import lpu.external_libs.nnPUSB
import lpu.external_libs.nnPUSB.nnPU_loss



LOG = logging.getLogger(__name__)

# import torch.optim

import lpu.constants
import lpu.datasets.LPUDataset
import lpu.datasets.dataset_utils
import lpu.utils.utils_general
import lpu.external_libs.nnPUSB.dataset
import lpu.models.nnPU


TRAIN_VAL_RATIO = .1
HOLD_OUT_SIZE = None
TRAIN_TEST_RATIO = .5


def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/nnPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False, invert_l=False)
    train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=len(lpu_dataset), hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    # nnPU_model = lpu.models.nnPU.nnPU(config)
    args = types.SimpleNamespace(**config)


    X_example, _, _ = next(iter(train_loader))
    dim = X_example.shape[-1]
    nnPU_model = lpu.models.nnPU.nnPU(config=config, dim=dim)
    nnPU_model.set_C(copy.deepcopy(train_loader))
    optimizer = torch.optim.Adam([{
        'params': nnPU_model.parameters(), 'lr': config.get('learning_rate')
    }])
    device = config.get('device', 'cpu')
    loss_func = lpu.models.nnPU.nnPUloss(prior=nnPU_model.prior, loss=lpu.external_libs.nnPUSB.train.select_loss('sigmoid'), gamma=args.gamma, beta=args.beta)
    for i in range(args.epoch):
        nnPU_model.train_one_epoch(dataloader=train_loader, optimizer=optimizer, loss_func=loss_func, device=device)
        print(i, nnPU_model.validate(val_loader))



if __name__ == "__main__":
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/nnPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    # lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False, invert_l=False)
    # train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=len(lpu_dataset), hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    # nnPU_model = lpu.models.nnPU.nnPU(config)
    args = types.SimpleNamespace(**config)

    with unittest.mock.patch('lpu.external_libs.nnPUSB.args.device', return_value=args.device):
        import lpu.external_libs.nnPUSB.train
        import lpu.external_libs.nnPUSB.model
        main()

