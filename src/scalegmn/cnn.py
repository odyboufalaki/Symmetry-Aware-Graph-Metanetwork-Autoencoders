import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.func import vmap

# class CNN(nn.Module):
#     def __init__(self, in_channels, num_classes, n_layers=3, n_hidden=16,
#                  dropout_rate=0.0, activation='relu', stride=2, use_batchnorm=False):
#         super(CNN, self).__init__()
        
#         self.layers = nn.ModuleList()
#         act_fn = {
#             'relu': nn.ReLU(),
#             'tanh': nn.Tanh(),
#             'sigmoid': nn.Sigmoid(),
#             'selu': nn.SELU()
#         }[activation]

#         for i in range(n_layers):
#             conv = nn.Conv2d(
#                 in_channels=in_channels if i == 0 else n_hidden,
#                 out_channels=n_hidden,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=0  # same as 'valid' in TF
#             )
#             self.layers.append(conv)

#             # if dropout_rate > 0.0:
#             #     self.layers.append(nn.Dropout2d(p=dropout_rate))

#             # if use_batchnorm:
#             #     self.layers.append(nn.BatchNorm2d(n_hidden))

#             self.layers.append(act_fn)

#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(n_hidden, num_classes)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         x = self.global_pool(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)




# def build_cnn_from_params(weights, biases, in_channels, num_classes, n_layers, n_hidden, **kwargs):
#     """
#     Creates a CNN instance and loads the provided weights and biases into it.

#     Args:
#         weights (list or tuple of torch.Tensor): The weight tensors for each layer.
#         biases (list or tuple of torch.Tensor): The bias tensors for each layer.
#         ... (other CNN constructor args)

#     Returns:
#         An instance of the CNN class with the loaded parameters.
#     """
#     # Instantiate the CNN model. Pass all necessary arguments.
#     model = CNN(
#         in_channels=in_channels,
#         num_classes=num_classes,
#         n_layers=n_layers,
#         n_hidden=n_hidden,
#         activation=kwargs.get('activation', 'relu'),
#     )

#     # The order here is CRUCIAL and must match the CNN's layer creation order.
#     # Your CNN has `n_layers` conv layers followed by 1 linear classifier.
    
#     # Check if the number of provided parameters matches the model's layers
#     if (len(weights) != n_layers + 1) or (len(biases) != n_layers + 1):
#         raise ValueError(
#             f"Parameter list length mismatch. Expected {n_layers + 1} weights and biases, "
#             f"but got {len(weights)} and {len(biases)}."
#         )

#     # Load weights and biases for the convolutional layers
#     param_idx = 0
#     for i in range(n_layers):
#         # Find the Conv2d layer in the ModuleList
#         conv_layer = model.layers[i]

#         # Ensure it's a Conv2d layer
#         if not isinstance(conv_layer, nn.Conv2d):
#             # Adjust index if dropout/other layers are present before conv
#             # For our current CNN, the structure is simpler.
#             # Because `use_batchnorm` is false, layers are [Conv, Act, Conv, Act, ...]
#             # The i-th conv layer is at index i*2 for your current CNN class.
#             conv_layer = model.layers[i*2] # Adjust index based on CNN structure

#         if not isinstance(conv_layer, nn.Conv2d):
#              raise TypeError(f"Expected a Conv2d layer at index {i*2}, but found {type(conv_layer)}")


#         # Reshape and assign weights and biases
#         conv_layer.weight.data = weights[param_idx].clone()
#         conv_layer.bias.data = biases[param_idx].clone().squeeze() # Biases need to be 1D
#         param_idx += 1

#     # Load weights and biases for the final linear classifier layer
#     model.classifier.weight.data = weights[param_idx].clone()
#     model.classifier.bias.data = biases[param_idx].clone().squeeze()
    
#     return model

# def unflatten_params(flat_params, layout, shapes):
#     """
#     Takes a flat vector of parameters and reshapes it into lists of
#     weight and bias tensors according to the provided layout and shapes.
    
#     Args:
#         flat_params (torch.Tensor): A single flat vector of all model parameters.
#         layout (pd.DataFrame): The layout dataframe from NFNZooDataset.
#         shapes (dict): A dictionary mapping varname to its original shape.
    
#     Returns:
#         A tuple of (weights, biases).
#     """
#     weights, biases = [], []
#     current_pos = 0
#     for _, row in layout.iterrows():
#         varname = row['varname']
#         shape = shapes[varname]
#         num_elements = np.prod(shape)
        
#         # Slice the flat tensor and reshape
#         param_slice = flat_params[current_pos : current_pos + num_elements]
#         reshaped_param = param_slice.view(shape)
        
#         current_pos += num_elements
        
#         if "kernel:0" in varname: # TensorFlow naming convention for weights
#             weights.append(reshaped_param)
#         elif "bias:0" in varname:
#             biases.append(reshaped_param)
            
#     return weights, biases


def unflatten_params_batch(flat_params_batch, layout, shapes):
    """
    Batch-aware version of unflatten_params, corrected for TF -> PyTorch conventions.
    
    Args:
        flat_params_batch (torch.Tensor): A batched flat vector of shape (B, total_params).
        layout (pd.DataFrame): The layout dataframe.
        shapes (dict): Dictionary mapping varname to its original shape.
    
    Returns:
        A tuple of (weights, biases), where each is a list of batched tensors in PyTorch format.
    """
    B = flat_params_batch.shape[0]
    weights, biases = [], []
    current_pos = 0
    
    for _, row in layout.iterrows():
        varname = row['varname']
        shape = shapes[varname]
        num_elements = np.prod(shape)
        
        # Slice the entire batch of flat parameters
        param_slice = flat_params_batch[:, current_pos : current_pos + num_elements]
        
        # Reshape, preserving the batch dimension
        reshaped_param = param_slice.view(B, *shape)
        
        current_pos += num_elements
        
        if "kernel:0" in varname:
            if reshaped_param.ndim == 5:  # Batched Conv2D weights: (B, H, W, C_in, C_out)
                # Permute from (B, H, W, C_in, C_out) to (B, C_out, C_in, H, W)
                # Original dims:  0, 1, 2, 3,    4
                # Target dims:    0, 4, 3,    1, 2
                weights.append(reshaped_param.permute(0, 4, 3, 1, 2))
            elif reshaped_param.ndim == 3: # Batched Linear weights: (B, in_features, out_features)
                # Transpose from (B, in, out) to (B, out, in)
                weights.append(reshaped_param.transpose(1, 2))
            else:
                 weights.append(reshaped_param)
                 
        elif "bias:0" in varname:
            # Squeeze out the singleton dimension if it exists (e.g., shape is -16)
            # The functional calls expect a 1D bias vector per model in the batch.
            if reshaped_param.ndim > 2:
                 biases.append(reshaped_param.squeeze(-1))
            else:
                 biases.append(reshaped_param)
            
    return weights, biases

def cnn_functional_forward(x, weights, biases, n_layers, stride=2, activation_fn=F.relu):
    """
    A functional, stateless version of the CNN forward pass.
    This function processes a SINGLE model's parameters on a BATCH of images.

    Args:
        x (torch.Tensor): A batch of images (e.g., shape [B_img, C, H, W]).
        weights (list of torch.Tensor): List of UNBATCHED weight tensors for one model.
        biases (list of torch.Tensor): List of UNBATCHED bias tensors for one model.
    """
    # Convolutional layers
    for i in range(n_layers):
        x = F.conv2d(x, weights[i], biases[i], stride=stride, padding=0)
        x = activation_fn(x)

    # Global pooling and classifier
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    
    # The last weight/bias pair is for the classifier
    x = F.linear(x, weights[-1], biases[-1])
    return x


def get_logits_for_batch(params_batch, image_batch, model_config):
    """
    The main workhorse. Evaluates a batch of models on a batch of images using vmap.

    Args:
        params_batch (tuple): A tuple (weights, biases), where each is a list of BATCHED tensors.
        image_batch (torch.Tensor): A batch of images to test on.
        model_config (dict): A dict with model hyperparameters like n_layers, stride.

    Returns:
        A tensor of logits with shape (B_models, B_images, num_classes).
    """
    weights, biases = params_batch
    
    # `vmap` maps the functional_forward over the batch of parameters.
    # `in_dims=(None, 0, 0)` means:
    #   - `None`: Don't map over the image_batch 'x'. Use the same image batch for all models.
    #   - `0`: Map over the first dimension of the `weights` list elements.
    #   - `0`: Map over the first dimension of the `biases` list elements.
    # We pass n_layers and stride as static arguments.
    n_layers = len(model_config['data']['layer_layout']) - 2
    stride =  2 # It is fixed for all CNNs
    activation_name = model_config['data']['activation_function']
    # Create a dictionary to map names to actual function objects.
    activation_map = {
        'relu': F.relu,
        'tanh': torch.tanh,
        'silu': F.silu,
    }

    activation_fn = activation_map.get(activation_name)
    if activation_fn is None:
        raise ValueError(f"Unsupported activation function: '{activation_name}'")

    # `torch.func.vmap` automatically handles the batch dimension for us efficiently
    # It will virtually "unstack" the batch of weights/biases and run the function for each.
    logits = vmap(
        lambda w, b: cnn_functional_forward(image_batch, w, b, n_layers=n_layers, stride=stride, activation_fn=activation_fn),
        in_dims=(0, 0)
    )(weights, biases)

    return logits