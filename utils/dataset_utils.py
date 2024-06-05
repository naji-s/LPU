from torch.utils.data import Sampler
import numpy as np
import sklearn.metrics

class StratifiedSampler(Sampler):
    """Stratified Sampling without replacement."""
    def __init__(self, labels):
        self.num_samples = len(labels)
        self.class_vector = np.array(labels).astype(int)
        self.indices = np.arange(self.num_samples)
        
    def __iter__(self):
        # Find the unique classes and their respective counts
        class_counts = np.bincount(self.class_vector)
        class_indices = [np.where(self.class_vector == i)[0] for i in range(len(class_counts))]
        
        # Calculate the number of items per class in each batch
        n_batches = self.num_samples // len(class_counts)
        indices = []
        for _ in range(n_batches):
            for class_idx in class_indices:
                indices.append(class_idx[np.random.randint(len(class_idx))])
        np.random.shuffle(indices)
        
        # In case the total number of samples isn't divisible by the number of classes,
        # randomly select the remaining indices
        remainder = self.num_samples - len(indices)
        if remainder:
            extra_indices = np.random.choice(self.indices, size=remainder, replace=False)
            indices = np.concatenate([indices, extra_indices])
        
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

def report_metrics(probs, vals, ests):
        """
        Helper function to compute and return metrics based on the given probabilities and values.
        """
        if set(np.unique(vals)).union(np.unique(vals)).issubset({-1, 1}):
            vals = (vals + 1) / 2
        
        metrics = {}
        for arr in [vals, ests] if ests is not None else [vals]:
            if all(arr):
                idx = np.random.randint(len(arr))
                arr[idx] = 0
            if not any(arr):
                idx = np.random.randint(len(arr))
                arr[idx] = 1

        if ests is not None:
            metrics['accuracy'] = sklearn.metrics.accuracy_score(vals, ests)
            metrics['precision'] = sklearn.metrics.precision_score(vals, ests)
            metrics['recall'] = sklearn.metrics.recall_score(vals, ests)
            metrics['f1'] = sklearn.metrics.f1_score(vals, ests)
        metrics['auc'] = sklearn.metrics.roc_auc_score(vals, probs)
        metrics['APS'] = sklearn.metrics.average_precision_score(vals, probs)
        metrics['ll'] = sklearn.metrics.log_loss(vals, probs)
        return metrics

import numpy as np
import sklearn.model_selection
import sklearn.datasets
import torch
import torch.utils.data 

import lpu.constants
import lpu.datasets
import lpu.datasets.LPUDataset
# import lpu.external_libs.Self_PU
# import lpu.external_libs.Self_PU.datasets
# import lpu.external_libs.distPU.dataTools.factory
# import lpu.external_libs.nnPUSB
# import lpu.external_libs.nnPUSB.dataset
import lpu.models
# import lpu.models.nnPUSB
# Assuming LPUDataset is defined as per previous corrections


def index_group_split(index_arr=None, ratios_dict=None, random_state=None, strat_arr=None):
    """
    Create stratified train and validation splits for a given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_val_ratio (float): The ratio of training data to validation data.
        batch_size (int): The batch size for the data loaders.
        hold_out_size (float): The ratio of hold-out data to training data. Defaults to 0.
        train_test_ratio (float): The ratio of testing data to training data. Defaults to 0.1.
        random_state (int): The random seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing the training, validation, and hold-out data loaders.
            The training data loader is used for training the model.
            The validation data loader is used for evaluating the model during training.
            The hold-out data loader is used for estimating a constant value (c) for Elkan & Noto (2008) algorithm.
            The testing data loader is used for evaluating the model after training.
    """
    # Combine 'l' and 'y' for stratification

    indices_dict = {}
    past_ratio = 1.
    new_ratio = None
    for key, ratio in sorted(ratios_dict.items(), key=lambda x: -x[1]):
        new_ratio = ratio / past_ratio
            
        if strat_arr is None:
            strat_arr_split = None
        else:
            strat_arr_split = strat_arr[index_arr]

        if not np.allclose(new_ratio, 1.):
            indices_dict[key], index_arr = sklearn.model_selection.train_test_split(
                index_arr,
                train_size=new_ratio,
                shuffle=True, 
                random_state=random_state,
                stratify=strat_arr_split
            )    
        else:
            indices_dict[key] = index_arr
        past_ratio = past_ratio - ratio        
    return indices_dict

def make_data_loader(dataset, batch_size, sampler=None, drop_last=False):
    if batch_size is None:
        batch_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last, pin_memory=True, num_workers=1, persistent_workers=True)
    return sampler, dataloader


def initialize_inducing_points(dataloader, inducing_points_size):
    # Assuming INDUCING_POINTS_SIZE is defined as an attribute or constant
    # This method selects a subset of the training data to be used as inducing points
    inducing_points = []
    for X_batch, _, _, _ in dataloader:
        inducing_points.append(X_batch)
        if len(inducing_points) >= inducing_points_size:
            break
    inducing_points = torch.cat(inducing_points, dim=0)
    inducing_points = inducing_points[:inducing_points_size]
    return inducing_points

class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified Sampling without replacement."""
    def __init__(self, labels):
        self.num_samples = len(labels)
        self.class_vector = np.array(labels).astype(int)
        self.indices = np.arange(self.num_samples)
        
    def __iter__(self):
        # Find the unique classes and their respective counts
        class_counts = np.bincount(self.class_vector)
        class_indices = [np.where(self.class_vector == i)[0] for i in range(len(class_counts))]
        
        # Calculate the number of items per class in each batch
        n_batches = self.num_samples // len(class_counts)
        indices = []
        for _ in range(n_batches):
            for class_idx in class_indices:
                indices.append(class_idx[np.random.randint(len(class_idx))])
        np.random.shuffle(indices)
        
        # In case the total number of samples isn't divisible by the number of classes,
        # randomly select the remaining indices
        remainder = self.num_samples - len(indices)
        if remainder:
            extra_indices = np.random.choice(self.indices, size=remainder, replace=False)
            indices = np.concatenate([indices, extra_indices])
        
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples



            
def LPUD_to_MPED(lpu_dataset, indices, data_generating_process='CC'):
    import lpu.external_libs.PU_learning.helper
    if data_generating_process == 'CC':
        # use all unlabeled data, which means we are using labeled data by removing the labels, to 
        # make the case-control assumption hold
        unlabeled_indices = indices
    elif data_generating_process == 'SB':
        # use only data where l=0, which under selection bias assumption is the same as l=0
        unlabeled_indices = indices[lpu_dataset.l[indices]==0]
    else:
        raise ValueError('data_generating_process must be one of "CC" or "SB"')
        

    p_indices = indices[lpu_dataset.l[indices]==1]


    PDataset = lpu.external_libs.PU_learning.helper.PosData(
        transform=lpu_dataset.transform, target_transform=None, data=lpu_dataset.X[p_indices], index=np.arange(len(p_indices)))
    UDataset = lpu.external_libs.PU_learning.helper.UnlabelData(
        transform=lpu_dataset.transform, target_transform=None, 
        pos_data=lpu_dataset.X[unlabeled_indices][lpu_dataset.y[unlabeled_indices]==1], 
        neg_data=lpu_dataset.X[unlabeled_indices][lpu_dataset.y[unlabeled_indices]==0],
        index=np.arange(len(unlabeled_indices)))
    
    dataset_dict = {
        'PDataset': PDataset,
        'UDataset': UDataset,
    }
    indices_dict = {
        'indices': indices,
        'p_indices': p_indices,
        'u_indices': unlabeled_indices,
    }
    return dataset_dict, indices_dict

def create_dataloaders_dict(config, target_transform=None, label_transform=None,  transform=None):
    dataloders_dict = {}
    samplers_dict = {}
    dataset_dict = {}
    dataloaders_dict = {}
    indices_dict = {}
    ratios_dict = config['ratios']
    if config['dataset_kind'] == 'LPU':
        lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name=config['dataset_name'])    
    elif config['dataset_kind'] == 'distPU':
        dataset_train, dataset_test = lpu.external_libs.distPU.dataTools.factory.create_dataset(config['dataset_name'], config['datapath'])
        X = np.concatenate([dataset_train.X, dataset_test.X], axis=0)
        Y = np.concatenate([dataset_train.Y, dataset_test.Y], axis=0)
        dataset = lpu.external_libs.distPU.dataTools.factory.BCDataset(X=X, Y=Y)
        pu_dataset = lpu.external_libs.distPU.dataTools.factory.create_pu_dataset(dataset, config['num_labeled'])
        l = pu_dataset.Y_PU
        X = pu_dataset.X_train
        Y = np.concatenate([pu_dataset.Y_train, np.ones(len(pu_dataset.Y_PU) - len(pu_dataset.Y_train))])
        lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(device=config['device'], data_dict={'X': X, 'l': l, 'y': Y}, transform=transform, target_transform=target_transform)
    elif config['dataset_kind'] == 'selfPU':
        if config['dataset_name'] == 'mnist':
            (trainX, trainY), (testX, testY) = lpu.external_libs.Self_PU.datasets.get_mnist()
            _trainY, _testY = lpu.external_libs.Self_PU.datasets.binarize_mnist_class(trainY, testY)

            # print ("SELF PACED TYPE:", config['self_paced_type'], vars(args))
            dataset_train_clean = lpu.external_libs.Self_PU.datasets.MNIST_Dataset(1000, 60000, 
                trainX, _trainY, testX, _testY, split='train', ids=[],
                increasing=config['increasing'], replacement=config['replacement'], mode=config['self_paced_type'], top = config['top'], type="clean", seed = lpu.constants.RANDOM_STATE)
            X = torch.concat([torch.tensor(dataset_train_clean.X), testX], dim=0)
            Y = torch.concat([torch.tensor(dataset_train_clean.T), torch.tensor(_testY)], dim=0)
            l = torch.concat([torch.tensor(dataset_train_clean.Y), torch.tensor(_testY)], dim=0)

            # l = (l + 1) / 2
            # Y = (Y + 1) / 2
            lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(device=config['device'], data_dict={'X': X, 'l': l, 'y': Y}, transform=transform, target_transform=target_transform)

        else:
            raise ValueError("")
    elif config['dataset_kind'] == 'nnPUSB':
        if config['dataset_name'] == 'mnist':
            data_dict = {}
            dataset_dict, prior, dim = lpu.models.nnPUSB.load_dataset(dataset_name=config['dataset_name'], n_labeled=200, n_unlabeled=69800, with_bias=True, resample_model=config["resample_model"])
        else:
            raise ValueError("")
            
            
    l = lpu_dataset.l.cpu().numpy()
    y = lpu_dataset.y.cpu().numpy()
    l_uniques = np.unique(l)
    y_uniques = np.unique(y)
    l_binary = (l == l_uniques[1]).astype(int)
    y_binary = (y == y_uniques[1]).astype(int)
    l_y_cat_transformed = l_binary * 2 + y_binary
    split_indices_dict = index_group_split(np.arange(len(l_y_cat_transformed)), ratios_dict=ratios_dict, random_state=lpu.constants.RANDOM_STATE, strat_arr=l_y_cat_transformed)
    if len(dataset_dict) == 0:
        dataset_dict = {split: None for split in split_indices_dict.keys()}
    for split in split_indices_dict.keys():
        if dataset_dict[split]:
            datas_dict = {'X': dataset_dict[split][0], 'l': dataset_dict[split][1], 'y': dataset_dict[split][2]}
            split_dataset = lpu.datasets.LPUDataset.LPUDataset(device=config['device'], data_dict=datas_dict, transform=transform, target_transform=target_transform, label_transform=label_transform)
            samplers_dict[split], dataloaders_dict[split] = make_data_loader(dataset=split_dataset, batch_size=config['batch_size'][split])
        else:
            # *** DO NOT DELETE *** for the normal case where we have a LPU dataset
            X, l, y, _ = lpu_dataset[split_indices_dict[split]]
            dataset_dict[split] = lpu.datasets.LPUDataset.LPUDataset(device=config['device'], data_dict={'X': X, 'l': l, 'y': y}, transform=transform, target_transform=target_transform, label_transform=label_transform)
            samplers_dict[split], dataloaders_dict[split] = make_data_loader(dataset=dataset_dict[split], batch_size=config['batch_size'][split])
    return dataloaders_dict


def one_zero_to_minus_one_one(arr):
    return (arr * 2) - 1

def minus_one_one_to_one_zero(arr):
    return (arr + 1) / 2

def binary_inverter(arr):
    return 1 - arr