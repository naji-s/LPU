import torch
import numpy as np
DTYPE = torch.float32
ROOT_PATH = '/Users/naji/phd_codebase'
EPSILON = 1e-12
BATCH_SIZE = 64
NUMPY_DTYPE = np.float64
# assert str(NUMPY_DTYPE).split('.')[1].split("'")[0] == str(DTYPE).split('.')[1].split("'")[0]