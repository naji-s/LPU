import logging
import unittest.mock
import sys
import types

sys.path.append('/Users/naji/phd_codebase/lpu/external_libs/nnPUSB')

import torch.optim
import lpu.constants
import lpu.datasets.dataset_utils
import lpu.utils.utils_general
import lpu.external_libs.nnPUSB.dataset
import lpu.models.uPU
import lpu.utils.plot_utils

torch.set_default_dtype(lpu.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    'learning_rate': 0.01,
    'epoch': 50,
    'gamma': 1.0,
    'beta': 0.0
}

LOG = lpu.utils.utils_general.configure_logger(__name__)

def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/uPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)

    dataloaders_dict = lpu.datasets.dataset_utils.create_dataloaders_dict(config)

    X_example, _, _, _ = next(iter(dataloaders_dict['train']))
    dim = X_example.shape[-1]
    uPU_model = lpu.models.uPU.uPU(config=config, dim=dim)
    uPU_model.set_C(dataloaders_dict['holdout'])

    optimizer = torch.optim.Adam([{
        'params': uPU_model.parameters(),
        'lr': config.get('learning_rate', DEFAULT_CONFIG.get('learning_rate', None) if USE_DEFAULT_CONFIG else None)
    }])

    device = config.get('device', 'cpu')
    loss_func = lpu.models.uPU.uPUloss(prior=uPU_model.prior,
                                       loss=lpu.external_libs.nnPUSB.train.select_loss('sigmoid'),
                                       gamma=config.get('gamma', DEFAULT_CONFIG.get('gamma', None) if USE_DEFAULT_CONFIG else None),
                                       beta=config.get('beta', DEFAULT_CONFIG.get('beta', None) if USE_DEFAULT_CONFIG else None))

    num_epochs = config.get('epoch', DEFAULT_CONFIG.get('epoch', None) if USE_DEFAULT_CONFIG else None)
    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}
    scores_dict = {split: {} for split in dataloaders_dict.keys()}

    for epoch in range(num_epochs):
        scores_dict_item = uPU_model.train_one_epoch(dataloader=dataloaders_dict['train'], optimizer=optimizer, loss_func=loss_func, device=device)
        scores_dict['train'].update(scores_dict_item)
        all_scores_dict['train']['epochs'].append(epoch)

        for split in ['val', 'test']:
            scores_dict_item = uPU_model.validate(dataloaders_dict[split])
            scores_dict[split].update(scores_dict_item)
            all_scores_dict[split]['epochs'].append(epoch)

        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

        LOG.info(f"Epoch {epoch}: {scores_dict}")

    lpu.utils.plot_utils.plot_scores(all_scores_dict, loss_type='overall_loss')

if __name__ == "__main__":
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/uPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    args = types.SimpleNamespace(**config)

    with unittest.mock.patch('lpu.external_libs.nnPUSB.args.device', return_value=args.device):
        import lpu.external_libs.nnPUSB.train
        import lpu.external_libs.nnPUSB.model
        main()