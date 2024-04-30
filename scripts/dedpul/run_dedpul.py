import logging
import sys
sys.path.append('lpu/external_libs/DEDPUL')

LOG = logging.getLogger(__name__)

# import torch.optim

import lpu.constants
import lpu.datasets.LPUDataset
import lpu.datasets.dataset_utils
import lpu.models.dedpul
import lpu.utils.utils_general

LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
TRAIN_VAL_RATIO = .1
HOLD_OUT_SIZE = None
TRAIN_TEST_RATIO = .5

def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/dedpul_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    config['kernel_mode'] = 2
    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', transform=None, invert_l=False)
    # train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.index_group_split(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE, hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    dataloaders_dict = lpu.datasets.dataset_utils.create_dataloaders_dict(config)
    train_nn_options = config['train_nn_options']
    train_nn_options['n_epochs'] = config['num_epochs']
    train_nn_options['batch_size'] = config['batch_size']['train']
    # if config['training_mode'] == 'standard':
    #     if config['bayes']:
    #         loss = lpu.external_libs.DEDPUL.NN_functions.d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function, bayes_weight)
    #     else:
    #         loss = lpu.external_libs.DEDPUL.NN_functions.d_loss_standard(batch_mix, batch_pos, discriminator, loss_function)

    # else:
    #     loss = lpu.external_libs.DEDPUL.NN_functions.d_loss_nnRE(batch_mix, batch_pos, discriminator, nnre_alpha, beta=beta, gamma=gamma,
    #                     loss_function=config['nn_options']['loss_function'])

    # passing X to initialize_inducing_points to extract the initial values of inducing points
    dedpul_model = lpu.models.dedpul.DEDPUL(config)
    dedpul_model.train(dataloaders_dict['train'], train_nn_options=train_nn_options)
    dedpul_model.set_C(dataloaders_dict['holdout'])
    print(dedpul_model.validate(dataloader=dataloaders_dict['test'], holdoutloader=dataloaders_dict['holdout']))
    
if __name__ == "__main__":
    main()

