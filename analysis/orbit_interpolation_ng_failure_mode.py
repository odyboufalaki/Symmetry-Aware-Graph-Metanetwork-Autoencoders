# --- START OF FILE orbit_interpolation_ng_failure_mode.py (TARGETED ONLY) ---

import argparse
import os
import pathlib
import sys
from functools import partial
from typing import Callable

import torch
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from tqdm import tqdm

from analysis.utils.utils import interpolate_batch, load_ground_truth_image, NUM_INTERPOLATION_SAMPLES
from src.neural_graphs.experiments.data import Batch
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.autoencoder import create_batch_wb
from src.utils.helpers import set_seed
from src.neural_graphs.nn.gnn import GNNForClassification

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
        old_cfg_splits_path = cfg.data.test.splits_path
        cfg.data.test.splits_path = os.path.join(cfg.latent_analysis.tmp_dir, "splits.json")
        print("Using temporary splits path:", cfg.data.test.splits_path)

    split_set = hydra.utils.instantiate(cfg.data.test)

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
        cfg.test.orbit.splits_path = old_cfg_splits_path

    return (net, loader) if return_model else loader

# --- AFTER (The Fix) ---
def load_inr_as_batch(inr_path: str, label: int, device: torch.device) -> Batch:
    sd = torch.load(inr_path, map_location="cpu")
    
    # 1. Load weights and biases as lists of 2D/1D tensors
    weights_list = [p for k, p in sd.items() if "weight" in k]
    biases_list = [p for k, p in sd.items() if "bias" in k]
    
    # 2. Permute the weights, like the original dataset loader
    weights_list = [w.permute(1, 0) for w in weights_list]
    
    # 3. Add the feature dimension, like the original dataset loader
    weights_list = [w.unsqueeze(-1) for w in weights_list]
    biases_list = [b.unsqueeze(-1) for b in biases_list]
    
    # 4. Add the batch dimension
    weights = [w.unsqueeze(0) for w in weights_list]
    biases = [b.unsqueeze(0) for b in biases_list]
    
    return Batch(weights=tuple(weights), biases=tuple(biases), label=torch.tensor([label])).to(device)

# --- Core Loss Functions (Unchanged) ---
def inr_loss(ground_truth_image: torch.Tensor, reconstructed_image: torch.Tensor) -> torch.Tensor:
    return mse_loss(reconstructed_image, ground_truth_image, reduction="none").mean(dim=list(range(1, reconstructed_image.dim())))

def inr_loss_batches(batch: Batch, loss_fn: Callable, device: torch.device, reconstructed: bool = False) -> torch.Tensor:
    weights_dev = [w.to(device) for w in batch.weights]
    biases_dev = [b.to(device) for b in batch.biases]
    imgs = test_inr(weights_dev, biases_dev, permuted_weights=reconstructed)
    return loss_fn(imgs)

def compute_loss_matrix(interpolated_batches: list[Batch], mnist_ground_truth_img: torch.Tensor, device: torch.device, reconstructed: bool = False) -> torch.Tensor:
    loss_matrix = []
    for interpolated_batch in interpolated_batches:
        loss = inr_loss_batches(batch=interpolated_batch, loss_fn=partial(inr_loss, mnist_ground_truth_img), device=device, reconstructed=reconstructed)
        loss_matrix.append(loss)
    return torch.stack(loss_matrix).permute(1, 0)

# --- Argument Parsing (pre-Hydra) ---
def get_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ckpt", type=str, required=True, help="Path to the Neural Graphs model checkpoint.")
    p.add_argument("--inr_a", type=str, required=True, help="Path to the first specific INR for interpolation.")
    p.add_argument("--inr_b", type=str, required=True, help="Path to the second specific INR for interpolation.")
    p.add_argument("--mnist_ground_truth_img", type=str, required=True, help="Path to the ground truth image for loss calculation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_matrices", action="store_true", help="Save the output loss matrices.")
    p.add_argument("--orbit_transformation", type=str, default="PD", help="Label for the orbit type used (for filenames).")
    p.add_argument("--dataset_path", type=str, default="data/mnist-inrs/", help="Path to the dataset directory.")
    p.add_argument("--split_path", type=str, default="data/mnist-inrs/mnist_splits.json,", help="Path to the dataset split file.")
    args, unknown = p.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    return args

args = get_args()
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="../src/neural_graphs/experiments/inr_classification/configs", config_name="base", version_base=None)
@torch.no_grad()
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Merge CLI args into Hydra config
    override = { "latent_analysis": {
        "ckpt_path": args.ckpt, "seed": args.seed, "save_matrices": args.save_matrices,
        "inr_a": args.inr_a, "inr_b": args.inr_b, "mnist_ground_truth_img": args.mnist_ground_truth_img,
        "transform_type": args.orbit_transformation,
    }}
    cfg = OmegaConf.merge(cfg, override)
    
    set_seed(cfg.latent_analysis.seed)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Running in Targeted Interpolation Mode for Neural Graphs ---")
    print(f"  INR A: {cfg.latent_analysis.inr_a}")
    print(f"  INR B: {cfg.latent_analysis.inr_b}")

    # Load Model and Ground Truth Image
    net, loader = load_orbit_dataset_and_model(
        cfg=cfg,
        device=device,
    )
    mnist_ground_truth_img = load_ground_truth_image(cfg.latent_analysis.mnist_ground_truth_img, device=device)

    # Load the two specific INRs
    wb_a = load_inr_as_batch(cfg.latent_analysis.inr_a, label=0, device=device)
    wb_b = load_inr_as_batch(cfg.latent_analysis.inr_b, label=0, device=device)
    
    # 1. Naive Interpolation
    print("Performing naive interpolation...")
    interpolated_naive = interpolate_batch(wb_a, wb_b, NUM_INTERPOLATION_SAMPLES)
    loss_matrix_naive = compute_loss_matrix(interpolated_naive, mnist_ground_truth_img, device, reconstructed=True)

    # 2. Re-Based Interpolation (using the NG model)
    print("Performing Neural Graphs-based interpolation...")
    w_rec_a, b_rec_a = create_batch_wb(net((wb_a.weights, wb_a.biases)))
    w_rec_b, b_rec_b = create_batch_wb(net((wb_b.weights, wb_b.biases)))
    
    wb_rec_a = Batch(weights=w_rec_a, biases=b_rec_a, label=wb_a.label)
    wb_rec_b = Batch(weights=w_rec_b, biases=b_rec_b, label=wb_b.label)
    
    interpolated_rebased = interpolate_batch(wb_rec_a, wb_rec_b, NUM_INTERPOLATION_SAMPLES)
    loss_matrix_rebased = compute_loss_matrix(interpolated_rebased, mnist_ground_truth_img, device)
    
    # Save matrices if requested
    if cfg.latent_analysis.save_matrices:
        output_dir = pathlib.Path("analysis/resources/interpolation/matrices")
        output_dir.mkdir(parents=True, exist_ok=True)
        method = "neural_graphs"
        inr_a_name = pathlib.Path(cfg.latent_analysis.inr_a).parent.parent.name
        inr_b_name = pathlib.Path(cfg.latent_analysis.inr_b).parent.parent.name
        
        base_name = f"TARGETED-{method}-{cfg.latent_analysis.transform_type}-{inr_a_name}-vs-{inr_b_name}"
        torch.save(loss_matrix_naive, output_dir / f"loss_matrix-naive-{base_name}.pt")
        torch.save(loss_matrix_rebased, output_dir / f"loss_matrix-rebased-{base_name}.pt")
        print(f"Saved targeted loss matrices to {output_dir}")

if __name__ == "__main__":
    main()