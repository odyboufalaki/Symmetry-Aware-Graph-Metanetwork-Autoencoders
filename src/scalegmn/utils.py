import torch
import torch_geometric
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import Subset
import torchvision
import random
import torchvision.transforms as transforms

class RemapAndMeanGrayscale(torch.nn.Module):
    """
    Custom transform that first remaps a [0, 1] tensor to [-1, 1] and then
    converts it to grayscale by taking the mean across the channel dimension.
    This exactly replicates the specified TensorFlow preprocessing logic and order.
    """
    def __init__(self, min_out=-1.0, max_out=1.0):
        super().__init__()
        self.min_out = min_out
        self.max_out = max_out

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor (Tensor): A Tensor of shape (C, H, W) in the [0, 1] range.
        Returns:
            Tensor: Grayscale image of shape (1, H, W) in the [-1, 1] range.
        """
        # 1. Remap the [0, 1] range to [-1, 1]. This is applied to the 3-channel tensor.
        remapped_tensor = self.min_out + img_tensor * (self.max_out - self.min_out)
        
        # 2. Take the mean across the channel dimension (dim=0).
        #    The input to this operation is the remapped 3-channel tensor.
        grayscale_tensor = remapped_tensor.mean(dim=0, keepdim=True)
        
        return grayscale_tensor

def get_cifar10_train_loader(batch_size=256, num_workers=4, num_images=10000):
    """
    Loads the CIFAR-10 train set and preprocesses it to EXACTLY match the
    specified TF logic: scale to [0,1], remap to [-1,1], THEN average to grayscale.
    """
    
    transform = transforms.Compose([
        # 1. Converts PIL Image [0, 255] to FloatTensor [0.0, 1.0] of shape (C, H, W).
        transforms.ToTensor(),
        
        # 2. Custom transform that combines the remapping and grayscale steps
        #    in the correct order.
        RemapAndMeanGrayscale(min_out=-1.0, max_out=1.0)
    ])

# 1. Load the FULL training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,         # <-- Key difference: load the training data
        download=True, 
        transform=transform
    )

    # 2. Generate a list of random indices
    total_size = len(full_train_dataset)
    if num_images > total_size:
        raise ValueError(f"Requested {num_images} images, but train set only has {total_size}.")

    generator = torch.Generator().manual_seed(42)
    
    # Generate a permutation of indices using our local, seeded generator.
    # This permutation will be IDENTICAL every time.
    indices = torch.randperm(total_size, generator=generator).tolist()
    
    # Select the first `num_images` from the shuffled list
    subset_indices = indices[:num_images]

    # 3. Create the Subset
    subset_dataset = Subset(full_train_dataset, subset_indices)

    # 4. Create the DataLoader for the subset
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset, 
        batch_size=batch_size,
        shuffle=True,  # Shuffle is good for training/validation sets
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Created a DataLoader with {len(subset_dataset)} random images from the training set.")
    return subset_loader

def get_cifar10_test_loader(batch_size=256, num_workers=4):
    """
    Loads the CIFAR-10 test set and preprocesses it to EXACTLY match the
    specified TF logic: scale to [0,1], remap to [-1,1], THEN average to grayscale.
    """
    
    transform = transforms.Compose([
        # 1. Converts PIL Image [0, 255] to FloatTensor [0.0, 1.0] of shape (C, H, W).
        transforms.ToTensor(),
        
        # 2. Custom transform that combines the remapping and grayscale steps
        #    in the correct order.
        RemapAndMeanGrayscale(min_out=-1.0, max_out=1.0)
    ])

    # TODO: Check the directory
    # Download and load the CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # Create the DataLoader
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers,
                                              pin_memory=True)
    
    return test_loader


def fast_nn_to_edge_index(layer_layout, device, dtype=torch.long):
    cum_layer_sizes = np.cumsum([0] + layer_layout)
    layer_indices = [
        torch.arange(cum_layer_sizes[i], cum_layer_sizes[i + 1], dtype=dtype)
        for i in range(len(cum_layer_sizes) - 1)
    ]
    edge_index = torch.cat(
        [
            torch.cartesian_prod(layer_indices[i], layer_indices[i + 1])
            for i in range(len(layer_indices) - 1)
        ],
        dim=0
    ).to(device).t()
    return edge_index


def graph_to_wb(
    edge_features,
    node_features,
    weights,
    biases
):
    new_weights = []
    new_biases = []
    cnt1, cnt2 = 0, weights[0].shape[1]
    for i, w in enumerate(weights):
        new_weights.append(edge_features[:, cnt1: cnt1+w.shape[1], cnt2: cnt2+w.shape[2]])
        cnt1 += w.shape[1]
        cnt2 += w.shape[2]
    cnt1 = weights[0].shape[1]
    for i, b in enumerate(biases):
        new_biases.append(node_features[:, cnt1: cnt1 + b.shape[1]])
        cnt1 += b.shape[1]
    return new_weights, new_biases


# replace the below with get_node_layer()
def get_nodes_at_layer(x, layer_idx, ptr, layer='hidden'):
    """
    TODO: this assumes fixed architecture for the input models.
    Modify this to handle different architectures. Maybe create and pass the l(i), l: V -> [L].
    Get hidden nodes from node_features
    """
    if layer == 'hidden':
        selected_nodes = range(layer_idx[1], layer_idx[-2])
    elif layer == 'first':
        selected_nodes = range(layer_idx[0], layer_idx[1])
    else:
        raise ValueError('Invalid layer type')

    selected_nodes = torch.tensor([m + p for p in ptr[:-1] for m in selected_nodes])
    mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    mask[selected_nodes] = 1
    return mask.unsqueeze(1)
