import numpy as np
import sklearn.model_selection
import sklearn.datasets
import torch
import torch.utils.data 

import lpu.constants

# Assuming LPUDataset is defined as per previous corrections

def get_mnist():
    """ 
    Fetches the MNIST dataset from OpenML and preprocesses it for training and testing.
    Helper function for running SelfPU and nnPU algorithms on the MNIST dataset. The original
    one does not work with new numpy and pandas versions.
    """
    mnist = sklearn.datasets.fetch_openml('mnist_784', data_home=".")

    x = np.array(mnist.data)  # Converting DataFrame to numpy array
    y = np.array(mnist.target)  # Ensuring y is also a numpy array
    y = y.astype(np.int32)  # Converting target to integer type

    # Reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)

def create_stratified_splits(dataset, train_val_ratio=0., batch_size=None, hold_out_size=0, train_test_ratio=.1, return_indices=False):
    """
    Create stratified train and validation splits for a given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_val_ratio (float): The ratio of training data to validation data.
        batch_size (int): The batch size for the data loaders.
        hold_out_size (float): The ratio of hold-out data to training data. Defaults to 0.

    Returns:
        tuple: A tuple containing the training, validation, and hold-out data loaders.
            The training data loader is used for training the model.
            The validation data loader is used for evaluating the model during training.
            The hold-out data loader is used for estimating a constant value (c) for Elkan & Noto (2008) algorithm.
    """
    train_dataloader = None
    val_dataloader = None
    hold_out_dataloader = None
    test_dataloader = None
    
    # Combine 'l' and 'y' for stratification
    l_y_cat_transformed = dataset.l.cpu().numpy() * 2 + dataset.y.cpu().numpy()

    if train_test_ratio:
        train_indices, test_indices = sklearn.model_selection.train_test_split(
            np.arange(len(dataset)),
            test_size=train_test_ratio,
            shuffle=True
        )
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Calculate indices for train and validation splits
    if hold_out_size:
        train_indices, hold_out_indices = sklearn.model_selection.train_test_split(
            train_indices,
            test_size=hold_out_size,
            shuffle=True
        )

    train_indices, val_indices = sklearn.model_selection.train_test_split(
        np.arange(len(dataset)),
        stratify=l_y_cat_transformed,
        test_size=train_val_ratio,
        shuffle=True
    )
    if hold_out_size:
        train_indices, hold_out_indices = sklearn.model_selection.train_test_split(
            train_indices,
            test_size=hold_out_size,
            shuffle=True
        )

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    if hold_out_size:
        hold_out_sampler = torch.utils.data.SubsetRandomSampler(hold_out_indices)
        hold_out_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=hold_out_sampler)

    if train_val_ratio:
        val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  sampler=val_sampler)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    if return_indices:
        return train_dataloader, val_dataloader, hold_out_dataloader, test_dataloader, train_indices, val_indices, hold_out_indices, test_indices
    else:
        return train_dataloader, test_dataloader, val_dataloader, hold_out_dataloader

def initialize_inducing_points(dataloader, inducing_points_size):
    # Assuming INDUCING_POINTS_SIZE is defined as an attribute or constant
    # This method selects a subset of the training data to be used as inducing points
    inducing_points = []
    for X_batch, _, _ in dataloader:
        inducing_points.append(X_batch)
        if len(inducing_points) >= inducing_points_size:
            break
    inducing_points = torch.cat(inducing_points, dim=0)
    inducing_points = inducing_points[:inducing_points_size]
    return inducing_points

# Usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset_name = "animal_no_animal"  # Example dataset name
# dataset = LPUDataset(dataset_name=dataset_name, device=device)

# train_loader, val_loader = create_stratified_splits(dataset, train_val_ratio=0.2, batch_size=BATCH_SIZE, device=device)

# Now you can use train_loader and val_loader in your training and validation loops.

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

def normalize_features(X):
    """
    Normalizes the dataset.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=lpu.constants.DTYPE)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X = (X - X_mean) / X_std
    return X

            