"""General utils for the interpolation experiment."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from src.data import dataset
from src.data.base_datasets import Batch
from src.scalegmn.autoencoder import get_autoencoder
from src.scalegmn.inr import INR
from src.utils.helpers import overwrite_conf
import torch
import torch.nn.functional as F
import numpy as np
from torch.func import vmap
from tqdm import tqdm
from typing import List
from torch_geometric.data import Data
# --- Import your project's specific functions and classes ---
from src.data.cifar10_dataset import CNNBatch, cnn_to_tg_data, pad_and_flatten_kernel
from src.scalegmn.cnn import unflatten_params_batch, get_logits_for_batch
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed, mask_input, mask_hidden, count_named_parameters
from .data_utils import get_node_types, get_edge_types
from src.scalegmn.utils import get_cifar10_test_loader

# Define this constant, perhaps at the top of your main script
NUM_INTERPOLATION_SAMPLES = 40
BATCH_SIZE = 64

# DONT USE THIS FILE

def mark_hidden_nodes(layer_layout) -> torch.Tensor:
    hidden_nodes = torch.tensor(
            [False for _ in range(layer_layout[0])] +
            [True for _ in range(sum(layer_layout[1:-1]))] +
            [False for _ in range(layer_layout[-1])]).unsqueeze(-1)
    return hidden_nodes

def mark_input_nodes(layer_layout) -> torch.Tensor:
    input_nodes = torch.tensor(
        [True for _ in range(layer_layout[0])] +
        [False for _ in range(sum(layer_layout[1:]))]).unsqueeze(-1)
    return input_nodes



def _transform_weights_biases(w, max_kernel_size, linear_as_conv=False):
    """
    Convolutional weights are 4D, and they are stored in the following
    order: [out_channels, in_channels, height, width]
    Linear weights are 2D, and they are stored in the following order:
    [out_features, in_features]

    1. We transpose the in_channels and out_channels dimensions in
    convolutions, and the in_features and out_features dimensions in linear
    layers
    2. We have a maximum HxW value, and pad the convolutional kernel with
    0s if necessary
    3. We flatten the height and width dimensions of the convolutional
    weights
    4. We unsqueeze the last dimension of weights and biases
    """
    if w.ndim == 1:
        w = w.unsqueeze(-1)
        return w

    w = w.transpose(0, 1)

    if linear_as_conv:
        if w.ndim == 2:
            w = w.unsqueeze(-1).unsqueeze(-1)
        w = pad_and_flatten_kernel(w, max_kernel_size)
    else:
        w = (
            pad_and_flatten_kernel(w, max_kernel_size)
            if w.ndim == 4
            else w.unsqueeze(-1)
        )

    return w


def create_graphs_from_params(config, cnn_batch: CNNBatch) -> List[Data]:
    """
    Converts an in-memory CNNBatch object into a list of torch_geometric.data.Data objects.
    This replicates the logic of __getitem__ for a pre-loaded batch.
    
    Args:
        self: The dataset instance, used to access config and helper methods.
        cnn_batch (CNNBatch): A batch of CNN parameters already in memory.
    
    Returns:
        List[Data]: A list of graph objects ready for a DataLoader.
    """
    max_kernel_size = (3,3)
    linear_as_conv=False
    flattening_method=None
    graph_data_list = []
    num_models = len(cnn_batch.y)

    # Loop through each model in the batch
    for idx in range(num_models):
        # --- 1. Extract parameters for the single model at index `idx` ---
        # Note: We don't need torch.from_numpy since the tensors are already torch.Tensor
        weights = [w[idx] for w in cnn_batch.weights]
        biases = [b[idx] for b in cnn_batch.biases]
        score = cnn_batch.y[idx].item()
        
        # Determine activation function based on config, as it's not stored in CNNBatch
        # This assumes all models in the batch have the same activation.
        activation_function = config['data']['activation_function']

        # --- 2. Replicate the logic from __getitem__ ---
        conv_mask = [1 if w.ndim == 4 else 0 for w in weights]
        layer_layout = [weights[0].shape[1]] + [v.shape[0] for v in biases]

        # Apply transformations
        transformed_weights = tuple([
            transform_weights_biases(w, max_kernel_size,
                                           linear_as_conv=linear_as_conv)
            for w in weights
        ])
        transformed_biases = tuple([
            transform_weights_biases(b, max_kernel_size,
                                           linear_as_conv=linear_as_conv)
            for b in biases
        ])

        if flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        equiv_on_hidden = mask_hidden(config)
        get_first_layer_mask = mask_input(config)

        node_pos_embed = config['data'].get('node_pos_embed', False)
        edge_pos_embed = config['data'].get('edge_pos_embed', False)
        
        if self.node_pos_embed:
            node2type = get_node_types(layer_layout)
        if self.edge_pos_embed:
            edge2type = get_edge_types(layer_layout)

        if equiv_on_hidden:
            hidden_nodes = mark_hidden_nodes(layer_layout)
        if get_first_layer_mask:
            first_layer_nodes = mark_input_nodes(layer_layout)

        # --- 3. Call the graph creation function ---
        graph_data = cnn_to_tg_data(
            transformed_weights,
            transformed_biases,
            conv_mask,
            config['scalegmn_args']['direction'],
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
            node2type=node2type if self.node_pos_embed else None,
            edge2type=edge2type if self.edge_pos_embed else None,
            mask_hidden=hidden_nodes if self.equiv_on_hidden else None,
            mask_first_layer=first_layer_nodes if self.get_first_layer_mask else None,
            sign_mask=activation_function == 'tanh'
        )
        
        graph_data_list.append(graph_data)
        
    return graph_data_list

# This function is now much simpler because your `dataset` class does the heavy lifting.
def cnn_batch_to_graph_loader(
    cnn_batch: CNNBatch,
    conf: dict,
) -> torch_geometric.loader.DataLoader:
    """
    Converts a batched CNNBatch object into a torch_geometric.loader.DataLoader
    that the ScaleGMN model can process.
    """
    # Initialize your dataset class in a special "from_preloaded" mode if possible,
    # or create a temporary one. This example assumes we can pass parameters directly.
    # The key is to bypass file I/O and create graph objects from in-memory tensors.
    graph_data_list = cnn_graph_dataset.create_graphs_from_params(
        cnn_batch, conf
    ) # YOU WILL NEED TO IMPLEMENT THIS HELPER in your dataset.py

    loader = torch_geometric.loader.DataLoader(
        graph_data_list,
        batch_size=conf["batch_size"],
        shuffle=False
    )
    return loader


@torch.no_grad()
def get_accuracy_matrix_for_interpolated_batch(
    interpolated_batches: List[CNNBatch],
    model_config: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes a matrix of accuracies for a list of interpolated CNN batches.
    This is an adaptation of your `evaluate` function's core logic.
    
    Args:
        interpolated_batches (List[CNNBatch]): A list where each element is a
            CNNBatch object representing all models at one interpolation step (alpha).
        model_config (dict): The main configuration dictionary.
        device (torch.device): GPU or CPU.

    Returns:
        torch.Tensor: A tensor of shape [num_models, NUM_INTERPOLATION_SAMPLES]
                      containing the accuracy of each model at each interpolation step.
    """
    cifar_loader = get_cifar10_test_loader(batch_size=model_config["batch_size"])
    num_models = len(interpolated_batches[0].y)
    num_interpolation_steps = len(interpolated_batches)
    
    # Shape: [num_models, num_interpolation_steps]
    accuracy_matrix = torch.zeros(num_models, num_interpolation_steps, device=device)
    total_samples = len(cifar_loader.dataset)

    # We iterate over the interpolation steps (alphas)
    for i, cnn_batch in enumerate(tqdm(interpolated_batches, desc="Evaluating Interpolation Steps")):
        params_on_device = (
            tuple(w.to(device) for w in cnn_batch.weights),
            tuple(b.to(device) for b in cnn_batch.biases)
        )
        
        # Accumulate correct predictions for this interpolation step
        correct_preds_at_alpha = torch.zeros(num_models, device=device)
        
        # Inner loop over CIFAR-10 test data
        for cifar_images, cifar_labels in cifar_loader:
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)
            
            # This is the magic function you provided!
            logits_batch = get_logits_for_batch(params_on_device, cifar_images, model_config)
            
            preds_batch = torch.argmax(logits_batch, dim=2) # Shape: (B_models, B_images)
            correct_preds_at_alpha += (preds_batch == cifar_labels.unsqueeze(0)).sum(dim=1)
            
        # Calculate accuracy for this alpha and store it in the matrix
        accuracy_matrix[:, i] = correct_preds_at_alpha / total_samples
        
    return accuracy_matrix.cpu() # Return on CPU


def perturb_cnn_batch(cnn_batch: CNNBatch, perturbation: float) -> CNNBatch:
    """Perturbs all weights and biases in a CNNBatch with Gaussian noise."""
    perturbed_weights = [
        w + perturbation * torch.randn_like(w) for w in cnn_batch.weights
    ]
    perturbed_biases = [
        b + perturbation * torch.randn_like(b) for b in cnn_batch.biases
    ]
    return CNNBatch(
        weights=perturbed_weights,
        biases=perturbed_biases,
        y=cnn_batch.y,
    )


def interpolate_cnn_batch(
    cnn_batch_1: CNNBatch,
    cnn_batch_2: CNNBatch,
) -> list[CNNBatch]:
    """Linearly interpolates between two CNNBatch objects."""
    alpha_values = torch.linspace(0, 1, NUM_INTERPOLATION_SAMPLES)
    interpolated_batches = []
    
    for alpha in alpha_values:
        interp_weights = [
            (1 - alpha) * w1 + alpha * w2 
            for w1, w2 in zip(cnn_batch_1.weights, cnn_batch_2.weights)
        ]
        interp_biases = [
            (1 - alpha) * b1 + alpha * b2 
            for b1, b2 in zip(cnn_batch_1.biases, cnn_batch_2.biases)
        ]
        interpolated_batches.append(
            CNNBatch(weights=interp_weights, biases=interp_biases, y=cnn_batch_1.y)
        )
        
    return interpolated_batches