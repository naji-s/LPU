
import copy
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
import lpu.utils.utils_general



    
LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
BATCH_SIZE = 32
HOLDOUT_RATIO = 0.05
VAL_RATIO = 0.10
TEST_RATIO = 0.35

    
def create_dataloaders_dict_mpe(config):
    dataloders_dict = {}
    samplers_dict = {}
    mpe_dataset_dict = {}
    mpe_dataloaders_dict = {}
    mpe_indices_dict = {}
    ratios_dict = config['ratios']
    if config['dataset_kind'] == 'LPU':
        lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')    
        l_y_cat_transformed = lpu_dataset.l.cpu().numpy() * 2 + lpu_dataset.y.cpu().numpy()
        split_indices_dict = lpu.datasets.dataset_utils.index_group_split(np.arange(len(l_y_cat_transformed)), ratios_dict=ratios_dict, random_state=lpu.constants.RANDOM_STATE, strat_arr=l_y_cat_transformed)
        for split in split_indices_dict.keys():
            # *** DO NOT DELETE *** for the normal case where we have a LPU dataset
            samplers_dict[split], dataloders_dict[split] = lpu.datasets.dataset_utils.make_data_loader(lpu_dataset, split_indices_dict[split], batch_size=BATCH_SIZE)

            mpe_dataset_dict[split], mpe_indices_dict[split] = lpu.datasets.dataset_utils.LPUD_to_MPED(lpu_dataset=lpu_dataset, indices=split_indices_dict[split], double_unlabeled=False)
            mpe_dataloaders_dict[split] = {}
            for dataset_type in mpe_dataset_dict[split].keys():
                mpe_dataloaders_dict[split][dataset_type] = torch.utils.data.DataLoader(mpe_dataset_dict[split][dataset_type], batch_size=config['batch_size'][split], shuffle=True)
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

def plot_scores(scores_dict, loss_type='L_mpe'):
    fig, ax = plt.subplots(5, 1, figsize=(10, 10))
    # Calculate the index of the minimum Total Loss
    min_loss_index = np.argmin(scores_dict['val'][loss_type])  # Index of minimum Total Loss

    # AUC Plot
    ax[0].plot(scores_dict['val']['y_auc'], label='val AUC')
    ax[0].plot(min_loss_index, scores_dict['val']['y_auc'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min Total Loss point
    ax[0].set_title('AUC')
    ax[0].legend()

    # Accuracy Plot
    ax[1].plot(scores_dict['val']['y_accuracy'], label='val Accuracy')
    ax[1].plot(min_loss_index, scores_dict['val']['y_accuracy'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min Total Loss point
    ax[1].set_title('val Accuracy')

    # Test AUC and Accuracy Plot
    ax[2].plot(scores_dict['test']['y_auc'], label='Test AUC')
    ax[2].plot(min_loss_index, scores_dict['test']['y_auc'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[2].set_title('Test AUC')

    ax[3].plot(scores_dict['test']['y_accuracy'], label='Test Accuracy')
    ax[3].plot(min_loss_index, scores_dict['test']['y_accuracy'][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[3].set_title(f'Test Accuracy')

    ax[4].plot(scores_dict['train'][loss_type], label=f'train {loss_type}')
    ax[4].plot(min_loss_index, scores_dict['train'][loss_type][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[4].set_title(f'train {loss_type}')

    ax[4].plot(scores_dict['val'][loss_type], label=f'val {loss_type}')
    ax[4].plot(min_loss_index, scores_dict['val'][loss_type][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[4].set_title(f'val {loss_type}')

    ax[4].plot(scores_dict['test'][loss_type], label=f'test {loss_type}')
    ax[4].plot(min_loss_index, scores_dict['test'][loss_type][min_loss_index], 'rx', markersize=10, label='Min Total Loss')  # Mark the min point
    ax[4].set_title(f'Test {loss_type}')


    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.legend()
    plt.show()


def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/mpe_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    
    criterion = torch.nn.CrossEntropyLoss()

    mpe_model = lpu.models.mpe_model.MPE(config)

    mpe_dataloaders_dict = create_dataloaders_dict_mpe(config)
    mpe_model.initialize_model(mpe_dataloaders_dict['train']['UDataset'].dataset.data.shape[1])
    train_unlabeled_size = len(mpe_dataloaders_dict['train']['UDataset'].dataset.data)


    if config['optimizer']=="SGD":
        optimizer = torch.optim.SGD(mpe_model.net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['wd'])
    elif config['optimizer']=="Adam":
        optimizer = torch.optim.Adam(mpe_model.net.parameters(), lr=config['lr'],weight_decay=config['wd'])
    elif config['optimizer']=="AdamW": 
        optimizer = torch.optim.AdamW(mpe_model.net.parameters(), lr=config['lr'])

    ## Train in the begining for warm start
    if config['warm_start']: 
        for epoch in range(config['warm_start_epochs']): 
            mpe_model.warm_up_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'], u_trainloader=mpe_dataloaders_dict['train']['UDataset'], optimizer=optimizer, criterion=criterion, valid_loader=None)
            if config['estimate_alpha']: 
                mpe_model.alpha_estimate = mpe_model.estimate_alpha(p_holdoutloader=mpe_dataloaders_dict['holdout']['PDataset'], u_holdoutloader=mpe_dataloaders_dict['holdout']['UDataset'])
                mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))
            print ("Epoch: ", epoch, "Alpha: ", mpe_model.alpha_estimate)

    scores_dict = {}
    for split in mpe_dataloaders_dict.keys():
        scores_dict[split] = {}
                   
    for epoch in range(config['epochs']):
        for split in mpe_dataloaders_dict.keys():
            scores_dict_item = mpe_model.validate(None, mpe_dataloaders_dict[split]['PDataset'], mpe_dataloaders_dict[split]['UDataset'], criterion=criterion, threshold=0.5)
            for score_type in scores_dict_item.keys():
                if score_type in scores_dict[split].keys():
                    scores_dict[split][score_type].append(scores_dict_item[score_type])
                else:
                    scores_dict[split][score_type] = [scores_dict_item[score_type]]

        mpe_model.train_one_epoch(epoch=epoch, p_trainloader=mpe_dataloaders_dict['train']['PDataset'], 
                       u_trainloader=mpe_dataloaders_dict['train']['UDataset'], optimizer=optimizer, criterion=criterion, 
                       train_unlabeled_size=train_unlabeled_size)

        if config['estimate_alpha']: 
            mpe_model.alpha_estimate = mpe_model.estimate_alpha(mpe_dataloaders_dict['holdout']['PDataset'], mpe_dataloaders_dict['holdout']['UDataset'])
            mpe_model.set_C(l_mean=len(mpe_dataloaders_dict['holdout']['PDataset']) / (len(mpe_dataloaders_dict['holdout']['UDataset']) + len(mpe_dataloaders_dict['holdout']['PDataset'])))

    for split in mpe_dataloaders_dict.keys():
        for score_type in scores_dict_item.keys():
            scores_dict[split][score_type] = np.asarray(scores_dict[split][score_type]).flatten()

    plot_scores(scores_dict)



if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", category=UserWarning)
    main()
