import torch
import numpy as np
DTYPE = torch.float64
ROOT_PATH = '/Users/naji/phd_codebase'
EPSILON = 1e-8
BATCH_SIZE = 64
NUMPY_DTYPE = np.float64
# assert str(NUMPY_DTYPE).split('.')[1].split("'")[0] == str(DTYPE).split('.')[1].split("'")[0]
RANDOM_STATE = 442
SGP_INDUCING_POINTS_SIZE = None
LOG_LEVEL = 'WARNING'
TUNING_RESULTS_DIR = f'{ROOT_PATH}/tuning_results'