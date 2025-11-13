
# --- START OF FILE orbit_interpolation_failure_mode.py (TARGETED ONLY) ---

import argparse
import os
import pathlib
from functools import partial
from typing import Callable

import torch
import yaml
from torch.nn.functional import mse_loss
from tqdm import tqdm

from analysis.linear_assignment import match_weights_biases_batch
from analysis.utils.utils import (
    interpolate_batch, load_ground_truth_image, remove_tmp_torch_geometric_loader,
    convert_and_prepare_weights, NUM_INTERPOLATION_SAMPLES
)
from analysis.utils.utils_sgmn import create_tmp_torch_geometric_loader, load_orbit_dataset_and_model
from src.data.base_datasets import Batch
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.autoencoder import create_batch_wb
from src.scalegmn.inr import INR
from src.utils.helpers import overwrite_conf, set_seed


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

# --- Main Function ---
def get_args():
    p = argparse.ArgumentParser(description="Targeted INR interpolation for ScaleGMN.")
    p.add_argument("--conf", type=str, required=True, help="YAML config for the ScaleGMN model.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to the ScaleGMN model checkpoint.")
    p.add_argument("--inr_a", type=str, required=True, help="Path to the first specific INR for interpolation.")
    p.add_argument("--inr_b", type=str, required=True, help="Path to the second specific INR for interpolation.")
    p.add_argument("--mnist_ground_truth_img", type=str, required=True, help="Path to the ground truth image for loss calculation.")
    p.add_argument("--tmp_dir", type=str, default="analysis/tmp_dir", help="Temporary directory for data loaders.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--linear_assignment", type=str, default=None, choices=["PD", "DP", "P", "D"], help="Optional: Use linear assignment instead of autoencoder.")
    p.add_argument("--save_matrices", action="store_true", help="Save the output loss matrices.")
    p.add_argument("--orbit_transformation", type=str, default="PD", help="Label for the orbit type used (for filenames).")
    p.add_argument("--dataset_path", type=str, default="data/mnist-inrs/", help="Path to the dataset directory.")
    p.add_argument("--split_path", type=str, default="data/mnist-inrs/mnist_splits.json", help="Path to the dataset split file.")
    return p.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Running in Targeted Interpolation Mode for ScaleGMN ---")
    print(f"  INR A: {args.inr_a}")
    print(f"  INR B: {args.inr_b}")

    # Load Model (without dataset) and Ground Truth Image
    net, _ = load_orbit_dataset_and_model(
        conf=conf,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ckpt_path=args.ckpt,
        device=device,
    )

    mnist_ground_truth_img = load_ground_truth_image(args.mnist_ground_truth_img, device=device)

    # Load the two specific INRs as both simple and PyG Batch objects
    wb_a = load_inr_as_batch(args.inr_a, label=0, device=device)
    wb_b = load_inr_as_batch(args.inr_b, label=0, device=device)
    
    inr_a_instance = INR(); inr_a_instance.load_state_dict(torch.load(args.inr_a))
    inr_b_instance = INR(); inr_b_instance.load_state_dict(torch.load(args.inr_b))
    
    loader_a = create_tmp_torch_geometric_loader([inr_a_instance], os.path.join(args.tmp_dir, "a"), conf, device)
    loader_b = create_tmp_torch_geometric_loader([inr_b_instance], os.path.join(args.tmp_dir, "b"), conf, device)
    
    batch_a, _ = next(iter(loader_a))
    batch_b, _ = next(iter(loader_b))

    # 1. Naive Interpolation
    print("Performing naive interpolation...")
    interpolated_naive = interpolate_batch(wb_a, wb_b, NUM_INTERPOLATION_SAMPLES)
    loss_matrix_naive = compute_loss_matrix(interpolated_naive, mnist_ground_truth_img, device, reconstructed=True)

    # 2. Re-Based Interpolation
    if args.linear_assignment:
        print(f"Performing Linear Assignment ('{args.linear_assignment}') interpolation...")
        rebased_w, rebased_b = match_weights_biases_batch(wb_a.weights, wb_b.weights, wb_a.biases, wb_b.biases, args.linear_assignment)
        rebased_w, rebased_b = convert_and_prepare_weights(rebased_w, rebased_b, device=device)
        wb_rebased = Batch(weights=rebased_w, biases=rebased_b, label=wb_b.label)
        interpolated_rebased = interpolate_batch(wb_a, wb_rebased, NUM_INTERPOLATION_SAMPLES)
    else:
        print("Performing Autoencoder-based interpolation...")
        w_rec_a, b_rec_a = create_batch_wb(net(batch_a.to(device)))
        w_rec_b, b_rec_b = create_batch_wb(net(batch_b.to(device)))
        wb_rec_a = Batch(weights=w_rec_a, biases=b_rec_a, label=wb_a.label)
        wb_rec_b = Batch(weights=w_rec_b, biases=b_rec_b, label=wb_b.label)
        interpolated_rebased = interpolate_batch(wb_rec_a, wb_rec_b, NUM_INTERPOLATION_SAMPLES)

    loss_matrix_rebased = compute_loss_matrix(interpolated_rebased, mnist_ground_truth_img, device)
    
    # Save matrices if requested
    if args.save_matrices:
        output_dir = pathlib.Path("analysis/resources/interpolation/matrices")
        output_dir.mkdir(parents=True, exist_ok=True)
        method = f"linear_assignment_{args.linear_assignment}" if args.linear_assignment else "scalegmn"
        inr_a_name = pathlib.Path(args.inr_a).parent.parent.name
        inr_b_name = pathlib.Path(args.inr_b).parent.parent.name
        
        base_name = f"TARGETED-{method}-{args.orbit_transformation}-{inr_a_name}-vs-{inr_b_name}"
        torch.save(loss_matrix_naive, output_dir / f"loss_matrix-naive-{base_name}.pt")
        torch.save(loss_matrix_rebased, output_dir / f"loss_matrix-rebased-{base_name}.pt")
        print(f"Saved targeted loss matrices to {output_dir}")

    remove_tmp_torch_geometric_loader(os.path.join(args.tmp_dir, "a"))
    remove_tmp_torch_geometric_loader(os.path.join(args.tmp_dir, "b"))

if __name__ == "__main__":
    main()