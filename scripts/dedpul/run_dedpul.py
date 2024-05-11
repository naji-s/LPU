import logging
import unittest.mock
import sys
sys.path.append('lpu/external_libs/DEDPUL')


import torch
import lpu.constants
import lpu.datasets.LPUDataset
import lpu.datasets.dataset_utils
import lpu.external_libs
import lpu.external_libs.DEDPUL
import lpu.external_libs.DEDPUL.NN_functions
import lpu.models.dedpul
import lpu.utils.plot_utils
import lpu.utils.utils_general



torch.set_default_dtype(lpu.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    'nrep': 1,
    'learning_rate': 1e-4,
    'n_neurons': 32,
    'MT_coef': 0.25,
    'bw_mix': 0.05,
    'bw_pos': 0.1,
    'threshold': 0.5,
    'num_epochs': 200,
    'batch_size': {
        'train': 64
    }
}

LOG = lpu.utils.utils_general.configure_logger(__name__)

def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/dedpul_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    config['kernel_mode'] = 2

    dataloaders_dict = lpu.datasets.dataset_utils.create_dataloaders_dict(config)

    train_nn_options = config['train_nn_options']
    train_nn_options['n_epochs'] = config.get('num_epochs', DEFAULT_CONFIG.get('num_epochs', None) if USE_DEFAULT_CONFIG else None)
    train_nn_options['batch_size'] = config.get('batch_size', DEFAULT_CONFIG.get('batch_size', None) if USE_DEFAULT_CONFIG else None)['train']

    dedpul_model = lpu.models.dedpul.DEDPUL(config)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}
    # for epoch in range(train_nn_options['n_epochs']):
    # scores_dict = {split: {} for split in dataloaders_dict.keys()}
    all_scores_dict.update(dedpul_model.train(train_dataloader=dataloaders_dict['train'], val_dataloader=dataloaders_dict['val'], test_dataloader=dataloaders_dict['test'], train_nn_options=train_nn_options))
    # scores_dict['train'].update(scores_dict_item)
    # all_scores_dict['train']['epochs'].append(epoch)

    # dedpul_model.set_C(dataloaders_dict['holdout'])
    # scores_dict_item = dedpul_model.validate(dataloader=dataloaders_dict['val'], holdoutloader=dataloaders_dict['holdout'])
    # scores_dict['val'].update(scores_dict_item)
    # all_scores_dict['val']['epochs'].append(epoch)

    # for split in dataloaders_dict.keys():
    #     for score_type, score_value in scores_dict[split].items():
    #         if score_type not in all_scores_dict[split]:
    #             all_scores_dict[split][score_type] = []
    #         all_scores_dict[split][score_type].append(score_value)

    LOG.info(f"{all_scores_dict}")
    # breakpoint()
    # lpu.utils.plot_utils.plot_scores(all_scores_dict, loss_type='overall_loss')

if __name__ == "__main__":
    main()