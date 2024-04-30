import numpy as np
import sklearn.model_selection
import torch
import torch.utils.data

import lpu.constants
import lpu.datasets.animal_no_animal.animal_no_animal_utils

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

class LPUDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading, preprocessing, and partitioning data 
    for learning purposes. This class is designed to be used with PyTorch
    """
    def __init__(self, data_dict=None, device=None, dataset_name=None,  transform=normalize_features, target_transform=None, load_all_data=True, invert_l=False):   
        """
        Initializes the dataset with data or loads from a specified dataset name. 

        Parameters:
            X: Input features.
            l: Whether a given example is labeled by an expert or not.
            device: The device on which the tensors will be allocated.
            dataset_name: Name of the dataset to load if X, l, and y are not provided.
        """        
        super().__init__()
        if data_dict is None:
            data_dict = {}
            index = None
        else:
            X, l, y, index = data_dict.get('X', None), data_dict.get('l', None), data_dict.get('y', None), data_dict.get('index', np.arange(len(data_dict['X'])))


        self.transform = transform
        self.target_transform = target_transform
        self.data_dict = data_dict
        self.device = device
        if not load_all_data:
            raise NotImplementedError("load_all_data=False is not supported yet. ")


        if dataset_name:
            X, y, l = self._read_data(dataset_name)
        if invert_l:
            l = 1 - l

        if index is not None:
            self._index = index            
        else:
            self._index = np.arange(len(X))

        X, l, y = X[self._index], l[self._index], y[self._index]

        X, l, y = self._check_input(X, l, y)
        if type(X) == np.ndarray:
            X, l, y = map(
                lambda arr: torch.tensor(arr, dtype=lpu.constants.DTYPE), [X, l, y])
        else:
            X, l, y = map(
                lambda arr: arr.to(lpu.constants.DTYPE), [X, l, y])
        # Normalize the input features
        if transform:
            X = self.transform(X)     

        if target_transform:
            y = self.target_transform(y)
            l = self.target_transform(l)

        self.X, self.l, self.y = X, l, y

    @property
    def set_index(self, index):
        """
        Sets the index of the dataset.

        Parameters:
        - index (numpy.ndarray): The new index to set.
        """
        self._index = index

    @property
    def get_index(self):
        """
        Returns the current index of the dataset.

        Returns:
        - numpy.ndarray: The current index of the dataset.
        """
        return self._index

    def __getitem__(self, idx):
        """
        Fetches and returns the data (X, l, y) for a given index after applying any necessary transformations.
        """
        return self.X[idx], self.l[idx], self.y[idx], self._index[idx]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.X)    

    def _check_input(self, X, l, y):
        """
        Validates the inputs X, l, and y for consistency in length, type, and ensures both y and l 
        are integer or float tensors, converting them to integer tensors if necessary.

        This function performs several checks:
        - Ensures that both y and l are not empty.
        - Verifies that X, l, and y all have the same length.
        - Checks that X, l, and y are of the same type.
        - For y and l:
            - If the first element is not an integer or if it is a tensor not of a floating-point dtype,
            they are converted to integer tensors.

        Parameters:
        - X (tensor or array-like): The input features or samples.
        - l (tensor or array-like): whether a given example is labeled by an expert or not.
        - y (tensor or array-like): The ttarget value, traditionally called y in a classification problem, associated with X.
        
        Returns:
        - tuple: A tuple containing X, l, and y after validation and potential type conversion.
        
        Raises:
        - ValueError: If y or l is empty, or if X, l, and y do not have the same length, or if they are not of the same type.
        """
        # Check if 'y' and 'l' are empty and raise an error if true.
        if not len(y):
            raise ValueError("y cannot be empty")
        if not len(l):
            raise ValueError("l cannot be empty")

        # Check if 'X', 'y', and 'l' have the same length and raise an error if they don't.
        if not (len(X) == len(y) == len(l)):
            raise ValueError("X, l, and y must have the same length")

        # Ensure 'X', 'l', and 'y' are of the same type, raising an error if not.
        if not (isinstance(X, type(y)) and isinstance(X, type(l))):
            raise ValueError("X, l, and y must be of the same type")

        # Validate 'y': if the first element is not an integer or if 'y' is a tensor not of floating point type,
        # convert 'y' to an integer tensor.
        if not isinstance(y[0], int) or (torch.is_tensor(y) and not y.dtype.is_floating_point):
            if torch.is_tensor(y):
                y = y.type(torch.int64)  # Convert 'y' to a 64-bit integer tensor.
            else:
                y = y.astype(int)  # Convert 'y' to an integer array if not a tensor.

        # Apply the same validation and conversion process to 'l' as was done for 'y'.
        if not isinstance(l[0], int) or (torch.is_tensor(l) and not l.dtype.is_floating_point):
            if torch.is_tensor(l):
                l = l.type(torch.int64)  # Convert 'l' to a 64-bit integer tensor.
            else:
                l = l.astype(int)  # Convert 'l' to an integer array if not a tensor.

        # Return the validated and potentially converted inputs.
        return X, l, y    

    def _read_data(self, dataset_name=None):
        """
        Reads the data from the hard drive and returns the input features, labels, and target values.
        """
        if dataset_name == 'animal_no_animal':
            output_location = f'{lpu.constants.ROOT_PATH}/datasets/animal_no_animal'
            subject = 'mta'
            model_type = 'vgg'  # or 'HMAX'
            layers_to_extract = ['classifier_0']
            X, y, l = lpu.datasets.animal_no_animal.animal_no_animal_utils.create_animal_no_animal_dataset(output_location=output_location, subject=subject, model_type=model_type, layers_to_extract=layers_to_extract)
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")            
        return X, y, l
    

