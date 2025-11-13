import json
import os
import numpy as np
#from src.scalegmn.models import ScaleGMN
import torch
import yaml
import torch_geometric
from tqdm import tqdm
#from src.utils.helpers import overwrite_conf
#from src.data import dataset
#from src.scalegmn.autoencoder import get_autoencoder
#from src.data.base_datasets import Batch
#from src.scalegmn.inr import INR
from omegaconf import DictConfig
from src.neural_graphs.nn.inr import INR
from src.neural_graphs.experiments.data import Batch
from src.neural_graphs.nn.gnn import GNNForClassification
from PIL import Image
import torchvision.transforms as transforms
from torch_geometric.utils import to_dense_adj
from src.neural_graphs.nn.gnn import to_pyg_batch
import matplotlib.pyplot as plt
import hydra




@torch.no_grad()
def collect_latents(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collects the latent codes from the model encoder for all samples in the dataset.
    Args:
        model (torch.nn.Module): The model to use for encoding.
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        device (torch.device): The device to use for computation.
    Returns:
        tuple: A tuple containing:
            - zs (torch.Tensor): The latent codes (tensor of shape [N, latent_dim]).
            - ys (torch.Tensor): The labels (tensor of shape [N]).
    """
    zs, ys = [], []
    model.eval()
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        inputs = (batch.weights, batch.biases)
        node_features, edge_features, _ = model.construct_graph(inputs)
        num_nodes = node_features.shape[1]
        labels = batch.label.cpu()
        batch = to_pyg_batch(node_features, edge_features, model.edge_index)
        out_node, out_edge = model.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
        )
        node_features = out_node.reshape(
            batch.num_graphs, num_nodes, out_node.shape[-1]
        )
        edge_features = to_dense_adj(batch.edge_index, batch.batch, out_edge)

        z = model.pool(node_features, edge_features)

        zs.append(z.cpu())
        ys.append(labels)
    return torch.cat(zs), torch.cat(ys)


def load_orbit_dataset_and_model(
    cfg: DictConfig,
    device: torch.device,
    tmp: bool = False,
    return_model: bool = True,
) -> tuple[GNNForClassification, torch.utils.data.DataLoader]:
    """
    Loads the orbit dataset and the pre-trained model.

    Args:
        conf (str): Path to the configuration YAML file.
        dataset_path (str): Path to the dataset directory.
        split_path (str): Path to the dataset split file.
        ckpt_path (str): Path to the model checkpoint file.
        device (torch.device): The device to load the model onto.
        debug (bool): If True, loads only a subset of the dataset for debugging.

    Returns:
        tuple: A tuple containing:
            - net (torch.nn.Module): The loaded model.
            - loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
    """
    if tmp:
        old_cfg_splits_path = cfg.data.orbit.splits_path
        cfg.data.orbit.splits_path = os.path.join(cfg.latent_analysis.tmp_dir, "splits.json")
        print("Using temporary splits path:", cfg.data.orbit.splits_path)
    #cfg.data.orbit.dataset_dir = cfg.latent_analysis.tmp_dir

    split_set = hydra.utils.instantiate(cfg.data.orbit)

    loader = torch.utils.data.DataLoader(
        dataset=split_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    print("Loaded", len(split_set), "samples in the dataset.")

  
    if return_model:
        point = split_set[0]
        weight_shapes = tuple(w.shape[:2] for w in point.weights)
        bias_shapes = tuple(b.shape[:1] for b in point.biases)

        layer_layout = [weight_shapes[0][0]] + [b[0] for b in bias_shapes]
        model_kwargs = dict()
        model_cls = cfg.model._target_.split(".")[-1]
        if model_cls == "DWSModelForClassification":
            model_kwargs["weight_shapes"] = weight_shapes
            model_kwargs["bias_shapes"] = bias_shapes
        else:
            model_kwargs["layer_layout"] = layer_layout
        net = hydra.utils.instantiate(cfg.model, **model_kwargs).to(device)
        checkpoint = torch.load(cfg.latent_analysis.ckpt_path, map_location=device)
        # Extract the model's state_dict from the checkpoint
        model_state_dict = checkpoint['model']
        # Now load the extracted state_dict into your network
        net.load_state_dict(model_state_dict)
        net.eval()
    
    if tmp:
        cfg.data.orbit.splits_path = old_cfg_splits_path

    return (net, loader) if return_model else loader


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
) -> list[Batch]:
    """Perturb the INR parameters in all batches of the dataset.

    Args:
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        perturbation (float): The amount to perturb the parameters by.

    Returns:
        list[Batch]: A list of perturbed batches of INR parameters.
    """
    perturbed_dataset = []
    for wb in loader:
        # Perturb weights and biases
        batch_perturbed = perturb_inr_batch(
            wb, perturbation=perturbation,
        )
        perturbed_dataset.append(batch_perturbed)
    return perturbed_dataset


def create_tmp_torch_geometric_loader(
    dataset: list[INR],
    tmp_dir: str,
    cfg: DictConfig,
    device: torch.device,
) -> torch.utils.data.DataLoader:
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
        cfg=cfg,
        device=device,
        tmp=True,
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

def plot_interpolation_curves(
    loss_matrices: list[tuple[torch.Tensor, str]],
    save_path: str = None,
) -> None:
    """Plot multiple interpolation curves.

    Args:
        loss_matrices (list[tuple[torch.Tensor, str]]): A list of tuples where each tuple contains:
            - A loss matrix of shape [BATCH_SIZE, NUM_INTERPOLATION_SAMPLES].
            - A string representing the name/label for the curve.
        save_path (str): Path to save the plot (optional).
    """
    plt.ylim(0, max([loss_matrix.mean(dim=0).max().item() for loss_matrix, _ in loss_matrices]) * 5)

    colors = plt.cm.viridis(np.linspace(0, 1, len(loss_matrices)))
    for color, (loss_matrix, label) in zip(colors, loss_matrices):
        loss_mean_curve = loss_matrix.mean(dim=0).cpu().numpy().astype(np.float64)
        loss_min_curve = loss_matrix.min(dim=0).values.cpu().numpy().astype(np.float64)
        loss_max_curve = loss_matrix.max(dim=0).values.cpu().numpy().astype(np.float64)
        print(f"Loss curve {label}: {loss_mean_curve}")
        print(f"Loss curve {label} min: {loss_min_curve}")
        print(f"Loss curve {label} max: {loss_max_curve}")
        plt.plot(
            np.arange(len(loss_mean_curve)) / (len(loss_mean_curve) - 1),
            loss_mean_curve,
            label=label,
            color=color,
        )
        plt.scatter(
            np.arange(len(loss_mean_curve)) / (len(loss_mean_curve) - 1),
            loss_mean_curve,
            marker='^',
            color=color,
        )
        plt.fill_between(
            np.arange(len(loss_mean_curve)) / (len(loss_mean_curve) - 1),
            loss_min_curve,
            loss_max_curve,
            alpha=0.3,
            label=f"{label} range (min-max)",
            color=color,
        )

    plt.xlabel("Interpolation Step")
    plt.ylabel("Loss")
    plt.title("Interpolation Loss Curves")
    plt.legend()
    

    if save_path:
        plt.savefig(save_path)
        print(f"Interpolation curves saved to {save_path}")
    
    
    plt.show()

    plt.close()
    
    