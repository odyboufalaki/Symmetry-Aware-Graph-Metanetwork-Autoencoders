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

NUM_INTERPOLATION_SAMPLES = 40
BATCH_SIZE = 64

def perturb_inr_batch(wb: Batch, perturbation: float) -> Batch:
    """Perturb the INR parameters in the batch by a small amount.

    Args:
        wb (Batch): The batch of INR parameters.
        perturbation (float): The amount to perturb the parameters by.

    Returns:
        Batch: The perturbed batch of INR parameters.
    """
    perturbed_weights = [
        stacked_tensors + perturbation * torch.randn_like(stacked_tensors)
        for stacked_tensors in wb.weights
    ]
    perturbed_biases = [
        stacked_tensors + perturbation * torch.randn_like(stacked_tensors)
        for stacked_tensors in wb.biases
    ]

    return Batch(
        weights=perturbed_weights,
        biases=perturbed_biases,
        label=wb.label,
    )


def perturb_inr_all_batches(
    loader: torch_geometric.loader.DataLoader,
    perturbation: float,
    is_tuple_loader: bool = True,
) -> list[Batch]:
    """Perturb the INR parameters in all batches of the dataset.

    Args:
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        perturbation (float): The amount to perturb the parameters by.
        is_tuple_loader (bool): Whether the loader returns tuples.
    Returns:
        list[Batch]: A list of perturbed batches of INR parameters.
    """
    perturbed_dataset = []
    for batch in loader:
        batch = batch[1] if is_tuple_loader else batch
        # Perturb weights and biases
        batch_perturbed = perturb_inr_batch(
            batch, perturbation=perturbation,
        )
        perturbed_dataset.append(batch_perturbed)
    return perturbed_dataset


def create_tmp_torch_geometric_loader(
    dataset: list[INR],
    tmp_dir: str,
    conf: dict,
    device: torch.device,
) -> torch_geometric.loader.DataLoader:
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    print(f"Creating temporary torch geometric loader from checkpoints in {tmp_dir}...")

    splits_json = dict()
    splits_json["test"] = {"path": [], "label": []}
    for inr_id, inr in enumerate(dataset):
        # Save the transformed INR to the output directory
        output_path_dir = os.path.join(
            tmp_dir,
            str(inr_id),
            "checkpoints",
        )
        os.makedirs(output_path_dir, exist_ok=True)

        output_path = os.path.join(
            output_path_dir,
            "model_final.pth",
        )
        # {"test": {"path": ["data/mnist-inrs/mnist
        splits_json["test"]["path"].append(output_path)
        splits_json["test"]["label"].append("2")
        torch.save(inr.state_dict(), output_path)

    # Save the splits JSON file
    with open(os.path.join(tmp_dir, "splits.json"), "w") as f:
        json.dump(splits_json, f, indent=4)

    # Load the dataset
    loader = load_orbit_dataset_and_model(
        conf=conf,
        dataset_path=tmp_dir,
        split_path=os.path.join(tmp_dir, "splits.json"),
        device=device,
        return_model=False,
    )

    return loader

    
def remove_tmp_torch_geometric_loader(
    tmp_dir: str
) -> None:
    print(f"Removing temporary directory {tmp_dir}...")
    # Remove the temporary directory
    for inr_id in range(len(os.listdir(tmp_dir))):
        parent_dir = os.path.join(tmp_dir, str(inr_id))
        output_path_dir = os.path.join(
            parent_dir,
            "checkpoints",
        )
        if os.path.exists(output_path_dir):
            for file in os.listdir(output_path_dir):
                os.remove(os.path.join(output_path_dir, file))
            os.rmdir(output_path_dir)        
            os.rmdir(parent_dir)
            
    if os.path.exists(os.path.join(tmp_dir, "splits.json")):
        os.remove(os.path.join(tmp_dir, "splits.json"))
    os.rmdir(tmp_dir)


def instantiate_inr_batch(
    batch: Batch,
    device: torch.device,
) -> list[INR]:
    """Instantiate an INR object from the batch of parameters.

    Args:
        wb (Batch): The batch of INR parameters.

    Returns:
        INR: The instantiated INR object.
    """
    ## INR State dict
    # seq.0.weight: torch.Size([32, 2])
    # seq.0.bias: torch.Size([32])
    # seq.1.weight: torch.Size([32, 32])
    # seq.1.bias: torch.Size([32])
    # seq.2.weight: torch.Size([1, 32])
    # seq.2.bias: torch.Size([1])

    dataset = []
    batch_size = len(batch.weights[0])
    for inr_id in range(batch_size):
        inr = INR()
        state_dict = inr.state_dict()
        for i, (weight, bias) in enumerate(zip(batch.weights, batch.biases)):
            state_dict[f"seq.{i}.weight"] = weight[inr_id].squeeze(-1).transpose(-1, 0)
            state_dict[f"seq.{i}.bias"] = bias[inr_id].squeeze(-1).transpose(-1, 0)
        inr.load_state_dict(state_dict)
        inr.eval()
        dataset.append(inr.to(device))
    
    return dataset


def instantiate_inr_all_batches(
    all_batches: list[Batch],
    device: torch.device,
) -> list[INR]:
    """
    Instantiate INRs from the dataset.

    Args:
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        device (torch.device): The device to use for computation.
    Returns:
        list[INR]: A list of instantiated INR objects.
    """
    dataset = []
    for wb in all_batches:
        dataset.extend(instantiate_inr_batch(
            batch=wb,
            device=device,
        ))
    return dataset


def interpolation_step_batch(
    inr_batch_1: Batch,
    inr_batch_2: Batch,
    alpha: float,
    interpolation_type: str = "linear",
) -> Batch:
    """Interpolate between two batches of INRs.

    Args:
        inr_batch_1 (Batch): The first batch of INR parameters.
        inr_batch_2 (Batch): The second batch of INR parameters.
        alpha (float): The interpolation factor (0 <= alpha <= 1).
        type (str): The type of interpolation to use.

    Returns:
        Batch: A list of interpolated INR objects.
    """
    # Interpolate between the two batches
    if interpolation_type == "linear":
        interpolated_weights = [
            (1 - alpha) * w1 + alpha * w2
            for w1, w2 in zip(inr_batch_1.weights, inr_batch_2.weights)
        ]
        interpolated_biases = [
            (1 - alpha) * b1 + alpha * b2
            for b1, b2 in zip(inr_batch_1.biases, inr_batch_2.biases)
        ]
    else:
        raise ValueError(f"Interpolation type {interpolation_type} not supported.")

    return Batch(
        weights=interpolated_weights,
        biases=interpolated_biases,
        label=inr_batch_1.label,
    )


def interpolate_batch(
    inr_batch_1: Batch,
    inr_batch_2: Batch,
    num_samples: int,
    interpolation_type: str = "linear",
) -> list[Batch]:
    """Interpolate between two batches of INRs.

    Args:
        inr_batch_1 (Batch): The first batch of INR parameters.
        inr_batch_2 (Batch): The second batch of INR parameters.
        num_samples (int): The number of samples to generate.
        interpolation_type (str): The type of interpolation to use.

    Returns:
        list[Batch]: A list of interpolated INR objects.
    """
    # Interpolate between the two batches
    if interpolation_type == "linear":
        alpha_values = torch.linspace(0, 1, num_samples)
    else:
        raise ValueError(f"Interpolation type {type} not supported.")

    interpolated_inrs = []
    for alpha in alpha_values:
        interpolated_inrs.append(
            interpolation_step_batch(inr_batch_1, inr_batch_2, alpha, interpolation_type)
        )

    return interpolated_inrs

def interpolate_latent_batch(
    latent_batch_1: list[torch.Tensor],
    latent_batch_2: list[torch.Tensor],
    num_samples: int,
    interpolation_type: str = "linear",
) -> list[Batch]:
    """Interpolate between two batches of latents."""
    if interpolation_type == "linear":
        alpha_values = torch.linspace(0, 1, num_samples)
    else:
        raise ValueError(f"Interpolation type {interpolation_type} not supported.")
    
    interpolated_latents = []
    for alpha in alpha_values:
        interpolated_latents.append(
            (1 - alpha) * latent_batch_1 + alpha * latent_batch_2
        )
    return interpolated_latents

def load_ground_truth_image(
    image_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Load the ground truth image from the specified path.

    Args:
        image_path (str): The path to the image file.
        device (torch.device): The device to load the image onto.

    Returns:
        torch.Tensor: The loaded image as a tensor.
    """
    # Open the image using PIL
    image = Image.open(image_path).convert("L")

    # Convert the image to a tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)

    return image_tensor


def convert_and_prepare_weights(rebased_weights, rebased_biases, device=None):
    """
    Convert rebased weights and biases to tensors and prepare them for _wb_to_tuple.
    
    Args:
        rebased_weights: List of lists of numpy arrays (weights for each layer of each INR)
        rebased_biases: List of lists of numpy arrays (biases for each layer of each INR)
        device: Optional torch.device to move tensors to
        
    Returns:
        Tuple of (weights, biases) in the format expected by _wb_to_tuple
    """
    # Convert to tensors
    weights_tensors = [
        [torch.from_numpy(w).to(device).float() if device else torch.from_numpy(w).float() 
         for w in inr_weights]
        for inr_weights in rebased_weights
    ]
    
    biases_tensors = [
        [torch.from_numpy(b).to(device).float() if device else torch.from_numpy(b).float() 
         for b in inr_biases]
        for inr_biases in rebased_biases
    ]
    
    # Reshape for _wb_to_tuple
    # We need to stack the weights and biases for each layer across the batch
    weights = [
        torch.stack([w[i] for w in weights_tensors]).unsqueeze(-1).permute(0,2,1,3)  # Add channel dimension
        for i in range(len(weights_tensors[0]))  # For each layer
    ]
    
    biases = [
        torch.stack([b[i] for b in biases_tensors]).unsqueeze(-1)  # Add channel dimension
        for i in range(len(biases_tensors[0]))  # For each layer
    ]
    
    return weights, biases


def plot_interpolation_curves(
    loss_matrices: list[tuple[torch.Tensor, str]],
    save_path: str = None,
    with_legend: bool = False
) -> None:
    """Plot multiple interpolation curves.

    Args:
        loss_matrices (list[tuple[torch.Tensor, str]]): A list of tuples where each tuple contains:
            - A loss matrix of shape [BATCH_SIZE, NUM_INTERPOLATION_SAMPLES].
            - A string representing the name/label for the curve.
        save_path (str): Path to save the plot (optional).
    """
    plt.figure(figsize=(6, 4))
    plt.ylim(0, max([loss_matrix.mean(dim=0).max().item() for loss_matrix, _ in loss_matrices]) * 1.5)

    colors = plt.cm.viridis(np.linspace(0, 1, len(loss_matrices)))
    for color, (loss_matrix, label) in zip(colors, loss_matrices):
        # Calculate statistics
        loss_mean_curve = loss_matrix.mean(dim=0).cpu().numpy().astype(np.float64)
        loss_std_curve = loss_matrix.std(dim=0).cpu().numpy().astype(np.float64)

        x_axis = np.arange(len(loss_mean_curve)) / (len(loss_mean_curve) - 1)
        # Plot mean curve
        plt.plot(
            x_axis,
            loss_mean_curve,
            '-',
            label=label,
            color=color,
            linewidth=1.5,
            zorder=3,
        )
        # Plot triangles at each point
        plt.scatter(
            x_axis,
            loss_mean_curve,
            marker='^',  # Triangle marker
            color=color,
            s=40,  # Size of markers
            zorder=4,  # Ensure markers are above the line
            alpha=0.7,  # Slightly transparent
        )
        
        # Plot std curve
        plt.fill_between(
            x_axis,
            loss_mean_curve - loss_std_curve,
            loss_mean_curve + loss_std_curve,
            color=color,
            alpha=0.2,
            zorder=1,
        )
       
    plt.tight_layout()
    plt.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.12)
    plt.xlabel("Interpolation Step")
    plt.ylabel("Loss")

    if with_legend:
        plt.legend(loc='upper left', fontsize=12, frameon=False)
    else:
        plt.legend().set_visible(False)

    if save_path:
        plt.savefig(save_path)
        print(f"Interpolation curves saved to {save_path}")
    plt.show()
    plt.close()
    