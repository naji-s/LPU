import logging
import unittest.mock
import sys
sys.path.append('/Users/naji/phd_codebase/lpu/external_libs/SAR_PU')
sys.path.append('/Users/naji/phd_codebase/lpu/external_libs/SAR_PU/sarpu')

import numpy as np
import torch
import lpu.constants
import lpu.datasets.LPUDataset
import lpu.external_libs.SAR_PU.sarpu.sarpu.PUmodels
import lpu.external_libs.SAR_PU.sarpu.sarpu.pu_learning
import lpu.external_libs.SAR_PU.sarpu
import lpu.external_libs.SAR_PU.sarpu.sarpu
import lpu.models.sarpu_em
import lpu.utils.utils_general
import lpu.datasets.dataset_utils
import lpu.utils.plot_utils

torch.set_default_dtype(lpu.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    'max_iter': 10
}

ORIGINAL_LOG_FUNC = np.log   
def stable_log(x):
    return ORIGINAL_LOG_FUNC(x + lpu.constants.EPSILON)

LOG = lpu.utils.utils_general.configure_logger(__name__)

def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/SARPU_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    max_iter = config.get('max_iter', DEFAULT_CONFIG.get('max_iter', None) if USE_DEFAULT_CONFIG else None)

    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')
    BATCH_SIZE = len(lpu_dataset)
    dataloaders_dict = lpu.datasets.dataset_utils.create_dataloaders_dict(config)

    sarpu_em_model = lpu.models.sarpu_em.SARPU(config, training_size=len(dataloaders_dict['train'].dataset))

    all_scores_dict = {split: {} for split in dataloaders_dict.keys()}

    scores_dict = {split: {} for split in dataloaders_dict.keys()}
    scores_dict_item = sarpu_em_model.train(dataloaders_dict['train'])
    scores_dict['train'].update(scores_dict_item)

    for split in ['val', 'test']:
        scores_dict_item = sarpu_em_model.validate(dataloaders_dict[split], loss_fn=sarpu_em_model.loss_fn)
        scores_dict[split].update(scores_dict_item)

    for split in dataloaders_dict.keys():
        all_scores_dict[split].update(scores_dict[split])

    LOG.info(f"Scores: {scores_dict}")
    # lpu.utils.plot_utils.plot_scores(all_scores_dict, loss_type='overall_loss')

if __name__ == "__main__":
    with unittest.mock.patch.object(np, 'log', stable_log) as mock_info:
        # since https://github.com/ML-KULeuven/SAR-PU/blob/c51f8af2b9604c363b6d129b7ad03b8db488346f/sarpu/sarpu/pu_learning.py#L162
        # can lead to unstable values (log of zero probs) we cchange the log function
        main()