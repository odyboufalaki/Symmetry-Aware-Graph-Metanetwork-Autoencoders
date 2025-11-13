"""Utils for the interpolation experiment with the ScaleGNN model."""
import json
import os

import torch
import torch_geometric
from tqdm import tqdm

from src.data import dataset
from src.scalegmn.autoencoder import get_autoencoder
from src.scalegmn.inr import INR
from src.scalegmn.models import ScaleGMN


@torch.no_grad()
def collect_latents(
    model: torch.nn.Module,
    loader: torch_geometric.loader.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Collects the latent codes from the model encoder for all samples in the dataset.
    Args:
        model (torch.nn.Module): The model to use for encoding.
        loader (torch_geometric.loader.DataLoader): The data loader for the dataset.
        device (torch.device): The device to use for computation.
    Returns:
        tuple: A tuple containing:
            - zs (torch.Tensor): The latent codes (tensor of shape [N, latent_dim]).
            - ys (torch.Tensor): The labels (tensor of shape [N]).
            - wbs (list[torch.Tensor]): The raw INR parameters (list of tensors).
    """
    zs, ys, wbs = [], [], []
    model.eval()
    for batch, wb in tqdm(loader, desc="Collecting latents"):
        batch = batch.to(device)
        z = model.encoder(batch)  # [B, latent_dim]
        zs.append(z.cpu())
        ys.append(batch.label.cpu())
        wbs.append(wb)  # raw INR params, useful for reconstructions
    return torch.cat(zs), torch.cat(ys), wbs


def load_orbit_dataset_and_model(
    conf: str,
    dataset_path: str,
    split_path: str,
    device: torch.device,
    ckpt_path: str = None,
    return_model: bool = True,
) -> tuple[ScaleGMN, torch_geometric.loader.DataLoader]:
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
    # Overwrite the dataset path
    conf["data"]["dataset_path"] = dataset_path

    # Load the orbit dataset
    conf["data"]["split_path"] = split_path
    split_set = dataset(
        conf["data"],
        split="test",
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=True,
        get_first_layer_mask=True,
        return_wb=True,
    )
    loader = torch_geometric.loader.DataLoader(
        split_set, batch_size=conf["batch_size"], shuffle=False,
    )
    print("Loaded", len(split_set), "samples in the dataset.")

    # Load the model
    decoder_hidden_dim_list = [
        conf["scalegmn_args"]["d_hid"] * elem
        for elem in conf["decoder_args"]["d_hidden"]
    ]
    conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list
    conf["scalegmn_args"]["layer_layout"] = split_set.get_layer_layout()
    
    if return_model:
        net = get_autoencoder(conf, autoencoder_type="inr").to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net.eval()

    return (net, loader) if return_model else loader


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