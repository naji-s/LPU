import torch
import numpy as np
DTYPE = torch.float64
ROOT_PATH = '/Users/naji/phd_codebase'
EPSILON = 1e-6
BATCH_SIZE = 64
NUMPY_DTYPE = np.float64
# assert str(NUMPY_DTYPE).split('.')[1].split("'")[0] == str(DTYPE).split('.')[1].split("'")[0]
RANDOM_STATE = 4442