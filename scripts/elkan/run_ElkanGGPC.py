import logging
import torch
import lpu.constants
import lpu.datasets.dataset_utils
import lpu.models.geometric.elkanGGPC
import lpu.utils.plot_utils
import lpu.utils.utils_general

torch.set_default_dtype(lpu.constants.DTYPE)

USE_DEFAULT_CONFIG = False
DEFAULT_CONFIG = {
    'inducing_points_size': 32,
    'learning_rate': 0.01,
    'epoch_block': 10  # Perform validation every EPOCH_BLOCK iterations
}

LOG = lpu.utils.utils_general.configure_logger(__name__)

def main():
    lpu.utils.utils_general.set_seed(lpu.constants.RANDOM_STATE)
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/psychm_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    
    inducing_points_size = config.get('inducing_points_size', DEFAULT_CONFIG.get('inducing_points_size', None) if USE_DEFAULT_CONFIG else None)
    dataloaders_dict = lpu.datasets.dataset_utils.create_dataloaders_dict(config)
    
    inducing_points_initial_vals = lpu.datasets.dataset_utils.initialize_inducing_points(
        dataloaders_dict['train'], inducing_points_size)

    elkan_model = lpu.models.geometric.elkanGGPC.ElkanGGPC(
        config, inducing_points_initial_vals=inducing_points_initial_vals,
        training_size=len(dataloaders_dict['train'].dataset),
        num_features=inducing_points_initial_vals.shape[-1]
    )
    
    learning_rate = config.get('learning_rate', DEFAULT_CONFIG.get('learning_rate', None) if USE_DEFAULT_CONFIG else None)
    num_epochs = config.get('num_epochs', DEFAULT_CONFIG.get('num_epochs', None) if USE_DEFAULT_CONFIG else None)
    epoch_block = config.get('epoch_block', DEFAULT_CONFIG.get('epoch_block', None) if USE_DEFAULT_CONFIG else None)
    optimizer = torch.optim.Adam([{
        'params': elkan_model.parameters(),
        'lr': learning_rate
    }])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    all_scores_dict = {split: {'epochs': []} for split in dataloaders_dict.keys()}

    for epoch in range(num_epochs):
        scores_dict = {split: {} for split in dataloaders_dict.keys()}

        scores_dict_item = elkan_model.train_one_epoch(optimizer=optimizer, dataloader=dataloaders_dict['train'], holdout_dataloader=dataloaders_dict['holdout'])
        scores_dict['train'].update(scores_dict_item)
        all_scores_dict['train']['epochs'].append(epoch)
        
        if epoch % epoch_block == 0:
            for split in ['val', 'test']:
                scores_dict_item = elkan_model.validate(dataloaders_dict[split], loss_fn=elkan_model.loss_fn)
                scores_dict[split].update(scores_dict_item)
                all_scores_dict[split]['epochs'].append(epoch)
            
            scheduler.step(scores_dict['val']['overall_loss'])
        
        for split in dataloaders_dict.keys():
            for score_type, score_value in scores_dict[split].items():
                if score_type not in all_scores_dict[split]:
                    all_scores_dict[split][score_type] = []
                all_scores_dict[split][score_type].append(score_value)
        
        LOG.info(f"Epoch {epoch}: {scores_dict}")
    
    lpu.utils.plot_utils.plot_scores(all_scores_dict, loss_type='overall_loss')

if __name__ == "__main__":
    main()