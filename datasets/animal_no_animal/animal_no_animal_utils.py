import sys, os
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
sys.path.append('/Users/naji/phd_codebase/psych_model/utils')
sys.path.append('/Users/naji/phd_codebase/psych_model/puLearning')
sys.path.append('/Users/naji/phd_codebase/psych_model/LPUModels')
sys.path.append('/Users/naji/phd_codebase/')
import glob
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from skimage.io import imread
import pickle
from lpu.extras.pytorch_hmax import hmax  # Ensure hmax is installed and accessible
import lpu.constants as constants
LOG = logging.getLogger(__name__)
# LOG.setLevel(logging.DEBUG)

# Define a custom dataset to load subject images
class SubjectDataset(Dataset):
    def __init__(self, images, model_type):
        self.images = images
        self.model_type = model_type
        self.transform = self.get_transform()

    def get_transform(self):
        if self.model_type == 'vgg':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif self.model_type == 'HMAX':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((224, 224)),  # Resize to match expected HMAX input size, adjust if necessary
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255),  # Scale pixel values as expected by HMAX
            ])
        else:
            raise NotImplementedError("Unsupported model type")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return self.transform(image)


def subject_related_data_read(output_location, subject, model_type):
    trial_data_location = f'{output_location}/subject_data/original/{subject}/'
    subject_related_response_dict = {"R": 1, "C": 0, "M": 0, "P": 1}
    Y_or_real_label_list, subject_related_response_list, subject_related_X_2d_RGB_list = [], [], []

    # Changed to process all '.exp' files instead of just the first one
    run_file_list = glob.glob(f'{trial_data_location}*.exp')

    for run_file_name in run_file_list:
        with open(run_file_name) as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:  # Ensure the line has enough parts to process
                        label, response = int(parts[4]), subject_related_response_dict.get(parts[3], -1)
                        X_location = 'animal' if parts[3] in ['M', 'R'] else 'no_animal'
                        image_path = f'{output_location}/pictures/{X_location}/{parts[2]}.jpg'
                        img = imread(image_path, pilmode='RGB' if model_type == 'vgg' else 'L')
                        Y_or_real_label_list.append(label)
                        subject_related_response_list.append(response)
                        subject_related_X_2d_RGB_list.append(img)

    return np.array(subject_related_X_2d_RGB_list), np.array(Y_or_real_label_list), np.array(subject_related_response_list)



def load_embeddings(embedding_file, model_type, layers_to_extract):
    """Load embeddings from file if they exist."""
    embeddings_dict = {}  # Initialize as an empty dictionary by default
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as f:
            loaded_dict = pickle.load(f)  # Load the file contents
            if isinstance(loaded_dict, dict):  # Check if the loaded content is a dictionary
                embeddings_dict = loaded_dict
    # Proceed with checking for missing and available layers
    missing_layers = [layer for layer in layers_to_extract if layer not in embeddings_dict.get(model_type, {})]
    available_layers = [layer for layer in layers_to_extract if layer in embeddings_dict.get(model_type, {})]
    return embeddings_dict, missing_layers, available_layers

def save_embeddings(embeddings_dict, embedding_file):
    """Save updated embeddings to file."""
    print(f"Saving embeddings to {embedding_file, embeddings_dict}")
    with open(embedding_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)

def update_embeddings(features, embeddings_dict, model_type, layers_to_extract):
    """Update embeddings dictionary with new features for missing layers."""
    if model_type not in embeddings_dict:
        embeddings_dict[model_type] = {}
    for layer in layers_to_extract:
        if layer in features:
            # Convert tensor to numpy array before storing
            embeddings_dict[model_type][layer] = features[layer].numpy()
    return embeddings_dict


def extract_and_save_features(model, dataloader, device, model_type, layers_to_extract, missing_layers, embeddings_dict, embedding_file):
    features = {}  # Use a dictionary to store features with layer names as keys

    with torch.no_grad():  # Ensure gradients are not calculated
        for inputs in dataloader:
            inputs = inputs.to(device)
            x = inputs  # Input to the model
            if model_type == 'vgg':
                # Process features block
                for idx, module in enumerate(model.features):
                    x = module(x)  # Forward pass through the layer
                    layer_name = f'features_{idx}'
                    if layer_name in layers_to_extract:  # Check if this layer's features need to be extracted
                        if layer_name not in features:
                            features[layer_name] = []
                        features[layer_name].append(x.detach().cpu())

                # Flatten the output for the classifier block
                x = torch.flatten(x, 1)

                # Process classifier block
                for idx, module in enumerate(model.classifier):
                    x = module(x)  # Forward pass through the layer
                    layer_name = f'classifier_{idx}'
                    if layer_name in layers_to_extract:  # Check if this layer's features need to be extracted
                        if layer_name not in features:
                            features[layer_name] = []
                        features[layer_name].append(x.detach().cpu())
            elif model_type == 'HMAX':
                for inputs_batch in inputs:
                    # Assuming inputs_batch is already a tensor and compatible with your HMAX model
                    s1_outputs, c1_outputs, s2_outputs, c2_outputs = model.run_all_layers(inputs_batch.unsqueeze(0))  # Add batch dimension if necessary
                    layer_outputs = {
                        'S1': s1_outputs,
                        'C1': c1_outputs,
                        'S2': torch.cat([item.unsqueeze(0) for sublist in s2_outputs for item in sublist]),  # Convert list of tensors to a single tensor
                        'C2': c2_outputs
                    }

                    # Iterate over the selected layers to extract
                    for layer_name in missing_layers:
                        if layer_name not in features:
                            features[layer_name] = []
                        # Now layer_outputs[layer_name] is guaranteed to be a tensor, so we can flatten it
                        features[layer_name].append(torch.flatten(layer_outputs[layer_name], start_dim=1).detach().cpu())       
            
    # Convert lists of tensors to single tensors
    for layer in features:
        features[layer] = torch.cat(features[layer], dim=0)  # Concatenate tensors along the batch dimension

    # Update embeddings dictionary with new features
    embeddings_dict = update_embeddings(features, embeddings_dict, model_type, missing_layers)

    # Save updated embeddings
    save_embeddings(embeddings_dict, embedding_file)

    return embeddings_dict


def get_embeddings_for_layers(embeddings_dict, model_type, layers_to_extract):
    """Concatenate embeddings for requested layers."""
    if model_type not in embeddings_dict:
        return np.array([])  # Return an empty array if the model type is not in the dictionary
    embeddings = [embeddings_dict[model_type].get(layer, np.array([])) for layer in layers_to_extract]
    if any(e.size == 0 for e in embeddings):  # Check if any layer is missing (has an empty array)
        return np.array([])  # Return an empty array if any layer's embeddings are missing
    # Ensure that all embeddings are 2D (samples, features) before concatenation
    embeddings = [e.reshape(e.shape[0], -1) for e in embeddings]  # Reshape each embedding to ensure it's 2D
    return np.concatenate(embeddings, axis=1)  # Concatenate along the feature axis


def extract_features_with_pytorch(output_location, subject, model_type='vgg', layers_to_extract=None):
    # Read subject related data from all .exp files
    subject_related_X_2d_RGB_arr, Y_or_real_label_arr, subject_related_response_list = subject_related_data_read(output_location, subject, model_type)

    # Initialize dataset and dataloader with the concatenated data
    dataset = SubjectDataset(subject_related_X_2d_RGB_arr, model_type)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)  # Adjust batch size as needed

    # Set device based on CUDA availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize model based on the specified model type
    if model_type == 'vgg':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device).eval()
        if layers_to_extract is None:
            layers_to_extract = [str(i) for i, _ in enumerate(model.features)]  # Use layer indices as strings
    elif model_type == 'HMAX':
        model = hmax.HMAX(f'{constants.ROOT_PATH}/pytorch_hmax/universal_patch_set.mat').to(device)
        if layers_to_extract is None:
            layers_to_extract = ['C2']  # Default to the final layer for HMAX
    else:
        raise NotImplementedError("Unsupported model type.")

    # LOG.info(f"Layers to be extracted are {layers_to_extract}")
    print(f"Layers to be extracted are {layers_to_extract}")
    # Define the file to store embeddings
    embedding_file = f'{output_location}/embeddings_{subject}_{model_type}.pkl'

    # Load existing embeddings if available
    embeddings_dict, missing_layers, available_layers = load_embeddings(embedding_file, model_type, layers_to_extract)

    # Extract and update embeddings for missing layers
    if missing_layers:
        embeddings_dict = extract_and_save_features(model, dataloader, device, model_type, layers_to_extract, missing_layers, embeddings_dict, embedding_file)
        # save_embeddings(embeddings_dict, embedding_file)  # Save updated embeddings

    # Retrieve embeddings for requested layers
    features = get_embeddings_for_layers(embeddings_dict=embeddings_dict, model_type=model_type, layers_to_extract=layers_to_extract)
    # Ensure features were successfully retrieved
    if features.size == 0 and layers_to_extract:
        raise ValueError("Failed to retrieve features for the requested layers.")

    # Return concatenated features along with labels and responses for all data
    return features, Y_or_real_label_arr, subject_related_response_list

def create_animal_no_animal_dataset(output_location, subject, model_type='vgg', layers_to_extract=None):
    features, labels, responses = extract_features_with_pytorch(output_location=output_location, subject=subject, model_type=model_type, layers_to_extract=layers_to_extract)
    X, y, r = features, labels, responses
    # Set responses to 1 for animal images
    # so false positive rate is set to 0
    r[np.logical_and(r == 1, y==1)] = 1
    r[np.logical_and(r == 1, y==0)] = 0
    return X, y, r
