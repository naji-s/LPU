import sys
import json
import importlib

import LPU.models.MPE.MPE
import LPU.utils.dataset_utils

MODEL_PATHS_DICT = {
    'MPE': 'LPU.models.MPE.MPE',
    'DEDPUL': 'LPU.models.DEDPUL.DEDPUL',
    'distPU': 'LPU.models.distPU.distPU',
    'nnPU': 'LPU.models.nnPU.nnPU',
    'nnPUSB': 'LPU.models.nnPUSB.nnPUSB',
    'sarpu': 'LPU.models.SARPU.SARPU',
    'selfPU': 'LPU.models.selfPU.selfPU',
    'TiCE': 'LPU.models.TiCE.TiCE',
    'uPU': 'LPU.models.uPU.uPU',
    'vpu': 'LPU.models.VPU.vPU',
    'KME': 'LPU.models.geometric.KME.KME',
    'PsychM': 'LPU.models.geometric.PsychM.PsychM',
    'Elkan': 'LPU.models.geometric.Elkan.Elkan',
    'Naive': 'LPU.models.geometric.Naive.Naive',

}

def get_data_loader_dict(config, model_name):
    data_loader_dict = None
    if model_name == 'MPE':
        data_loader_dict = LPU.models.MPE.MPE.create_dataloaders_dict_mpe(config)
    else:
        data_loader_dict = LPU.utils.dataset_utils.create_dataloaders_dict(config)
    return data_loader_dict
        
def load_model(model_name, config):
    model_module = importlib.import_module(MODEL_PATHS_DICT[model_name])
    breakpoint()
    model = getattr(model_module, f'{model_name}')(config)


def load_config(config=None, model_name=None, file_path=None):
    if config is None:
        config = {}
    # Load the base configuration
    module = importlib.import_module(f'LPU.scripts.{model_name}.run_{model_name}')
    model_default_config = getattr(module, 'DEFAULT_CONFIG')
    config = LPU.utils.utils_general.deep_update(model_default_config, config)
    return config

def train_model(model_name=None, config=None):    
    config = load_config(config, model_name)
    if 'random_state' in config and config['random_state'] is not None:
        random_state = config['random_state']
        # setting the seed for the training
        LPU.utils.utils_general.set_seed(random_state)


    data_loader_dict = get_data_loader_dict(config, model_name=model_name)
    model = load_model(model_name, config)
    # Assume train_model is a function that trains the model
    results = train_model(model, data_loader_dict, config)
    print(f"Training results for {model_name}: {results}")

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python train.py config_path model_name")
    #     sys.exit(1)
    # config_path, model_name = sys.argv[1], sys.argv[2]
    train_model(model_name=sys.argv[1])
