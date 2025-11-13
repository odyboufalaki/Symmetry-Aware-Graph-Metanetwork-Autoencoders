"""Utils for the interpolation experiment with the Neural Graphs model."""
import json
import os

import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.utils import to_dense_adj

from src.neural_graphs.nn.gnn import GNNForClassification, to_pyg_batch
from src.neural_graphs.nn.inr import INR


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