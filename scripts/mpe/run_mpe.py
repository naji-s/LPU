
import sys

sys.path.append('lpu/external_libs/PU_learning')
sys.path.append('lpu/external_libs/PU_learning/data_helper')
import lpu.external_libs.PU_learning.helper
import lpu.external_libs.PU_learning.utils

from matplotlib import pyplot as plt
import torch.nn
import torch.utils.data
import numpy as np

import lpu.constants
import lpu.datasets.dataset_utils
import lpu.datasets.LPUDataset
import lpu.external_libs.PU_learning.algorithm
import lpu.models.mpe_model
import lpu.utils.plot_utils
import lpu.utils.utils_general

        
def create_dataloaders_dict_mpe(config, drop_last=False):
    # dataloders_dict = {}
    # samplers_dict = {}
    mpe_dataset_dict = {}
    mpe_dataloaders_dict = {}
    mpe_indices_dict = {}
    ratios_dict = config['ratios']
    data_generating_process = config['data_generating_process']
    data_type = lpu.constants.DTYPE
    if config['dataset_kind'] == 'LPU':
        lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')    
        l_y_cat_transformed = lpu_dataset.l.cpu().numpy() * 2 + lpu_dataset.y.cpu().numpy()
        split_indices_dict = lpu.datasets.dataset_utils.index_group_split(np.arange(len(l_y_cat_transformed)), ratios_dict=ratios_dict, random_state=lpu.constants.RANDOM_STATE, strat_arr=l_y_cat_transformed)
        for split in split_indices_dict.keys():
            # *** DO NOT DELETE *** for the normal case where we have a LPU dataset
            # samplers_dict[split], dataloders_dict[split] = lpu.datasets.dataset_utils.make_data_loader(lpu_dataset, batch_size=config['batch_size'][split],)
            mpe_dataset_dict[split], mpe_indices_dict[split] = lpu.datasets.dataset_utils.LPUD_to_MPED(lpu_dataset=lpu_dataset, indices=split_indices_dict[split], data_generating_process=data_generating_process)
            mpe_dataloaders_dict[split] = {}
            for dataset_type in mpe_dataset_dict[split].keys():
                mpe_dataloaders_dict[split][dataset_type] = torch.utils.data.DataLoader(mpe_dataset_dict[split][dataset_type], batch_size=config['batch_size'][split], drop_last=drop_last, shuffle=True)
    elif config['dataset_kind'] == 'MPE':
        p_trainloader, u_trainloader, p_validloader, u_validloader, net, X, Y, p_validdata, u_validdata, u_traindata, p_traindata = \
                lpu.external_libs.PU_learning.helper.get_dataset(config['data_dir'], config['data_type'], config['net_type'], config['device'], config['alpha'], config['beta'], config['batch_size'])


        mpe_dataloaders_dict['train']= {}
        mpe_dataloaders_dict['test'] ={}
        mpe_dataloaders_dict['val'] = {}
        mpe_dataloaders_dict['holdout'] = {}

        mpe_dataloaders_dict['train']['PDataset'] = p_trainloader
        mpe_dataloaders_dict['train']['UDataset'] = u_trainloader

        mpe_dataloaders_dict['test']['PDataset'] = p_validloader
        mpe_dataloaders_dict['test']['UDataset'] = u_validloader

        mpe_dataloaders_dict['holdout']['PDataset'] = p_validloader
        mpe_dataloaders_dict['holdout']['UDataset'] = u_validloader


        mpe_dataloaders_dict['val']['PDataset'] = p_validloader
        mpe_dataloaders_dict['val']['UDataset'] = u_validloader
    else:
        raise ValueError("Dataset needs to be either LPU or MPE")
    return mpe_dataloaders_dict

LOG = lpu.utils.utils_general.configure_logger(__name__)


def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/mpe_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)

    criterion = torch.nn.CrossEntropyLoss()

    mpe_model = lpu.models.mpe_model.MPE(config)

    mpe_dataloaders_dict = create_dataloaders_dict_mpe(config)
    mpe_model.initialize_model(mpe_dataloaders_dict['train']['UDataset'].dataset.data.shape[1])
    train_unlabeled_size = len(mpe_dataloaders_dict['train']['UDataset'].dataset.data)

    if config['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(mpe_model.net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['wd'])
    elif config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(mpe_model.net.parameters(), lr=config['lr'], weight_decay=config['wd'])
    elif config['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(mpe_model.net.parameters(), lr=config['lr'])

    scores_dict = {}
    all_scores_dict = {split: {'epochs': []} for split in mpe_dataloaders_dict.keys()}

    ## Train in the beginning for warm start
    if config['warm_start']:
        for epoch in range(config['warm_start_epochs']):
            train_scores = mpe_model.warm_up_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'],
                                                       u_trainloader=mpe_dataloaders_dict['train']['UDataset'],
                                                       optimizer=optimizer, criterion=criterion, valid_loader=None)
            scores_dict['train'] = train_scores
            all_scores_dict['train']['epochs'].append(epoch)

            for split in ['val', 'test', 'holdout']:
                valid_scores = mpe_model.validate(epoch, p_validloader=mpe_dataloaders_dict[split]['PDataset'],
                                                  u_validloader=mpe_dataloaders_dict[split]['UDataset'],
                                                  criterion=criterion, threshold=0.5)
                scores_dict[split] = valid_scores
                all_scores_dict[split]['epochs'].append(epoch)

            if config['estimate_alpha']:
                mpe_model.alpha_estimate = mpe_model.estimate_alpha(p_holdoutloader=mpe_dataloaders_dict['holdout']['PDataset'],
                                                                    u_holdoutloader=mpe_dataloaders_dict['holdout']['UDataset'])
                mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))
            # print("Epoch: ", epoch, "Alpha: ", mpe_model.alpha_estimate)
            LOG.info(f"Warmup Epoch {epoch}: {scores_dict}")

            for split in mpe_dataloaders_dict.keys():
                for score_type, score_value in scores_dict[split].items():
                    if score_type not in all_scores_dict[split]:
                        all_scores_dict[split][score_type] = []
                    all_scores_dict[split][score_type].append(score_value)

    for epoch in range(config['epochs']):
        train_scores = mpe_model.train_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'],
                                                 u_trainloader=mpe_dataloaders_dict['train']['UDataset'],
                                                 optimizer=optimizer, criterion=criterion,
                                                 train_unlabeled_size=train_unlabeled_size)
        scores_dict['train'] = train_scores
        all_scores_dict['train']['epochs'].append(epoch + config['warm_start_epochs'])

        for split in ['val', 'test', 'holdout']:
            valid_scores = mpe_model.validate(epoch, p_validloader=mpe_dataloaders_dict[split]['PDataset'],
                                              u_validloader=mpe_dataloaders_dict[split]['UDataset'],
                                              criterion=criterion, threshold=0.5)
            scores_dict[split] = valid_scores
            all_scores_dict[split]['epochs'].append(epoch + config['warm_start_epochs'])

        if config['estimate_alpha']:
            mpe_model.alpha_estimate = mpe_model.estimate_alpha(mpe_dataloaders_dict['holdout']['PDataset'], mpe_dataloaders_dict['holdout']['UDataset'])
            mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))
        LOG.info(f"Train Epoch {epoch}: {scores_dict}")

        for split in mpe_dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)

    for split in mpe_dataloaders_dict.keys():
        for score_type, score_values in all_scores_dict[split].items():
            if score_type != 'epochs':
                all_scores_dict[split][score_type] = np.array(score_values)
    lpu.utils.plot_utils.plot_scores(all_scores_dict, loss_type='total_loss')



if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    main()
