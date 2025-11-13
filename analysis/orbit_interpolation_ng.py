"""
Orbit interpolation experiment for the Neural Graphs model.

A run of the interpolation experiment consists of:
1. Randomly choosing an INR from the test set.
2. Creating an orbit dataset of the INR.
3. Perturbing the INR parameters with a gaussian distributed noise (plus a perturbation parameter).
4. Performing interpolation in weights space:
    4.1 Naive interpolation: Between the original orbit INRs and the group-acted and perturbed INRs.
    4.2 Autoencoder interpolation: Between the reconstructed INRs of the original orbit INRs and the reconstructed INRs of the group-acted and perturbed INRs.
    NOTE: Reconstructing an INR implies passing it through the autoencoder.
5. Computing the loss matrices for all the previous interpolations.
6. Plotting the interpolation curves.

The interpolation experiment consists of multiple runs, where each run is done with a different INR.
"""
import argparse
import gc
import json
import os
import pathlib
import random
import sys
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from tqdm import tqdm

from analysis.utils.orbit_dataset import (
    delete_orbit_dataset,
    generate_orbit_dataset,
)
from analysis.utils.utils import (
    instantiate_inr_all_batches,
    interpolate_batch,
    load_ground_truth_image,
    perturb_inr_all_batches,
    remove_tmp_torch_geometric_loader,
    NUM_INTERPOLATION_SAMPLES,
    BATCH_SIZE,
    interpolate_latent_batch,
)
from analysis.utils.utils_ng import (
    create_tmp_torch_geometric_loader,
    load_orbit_dataset_and_model,
)
from src.neural_graphs.experiments.data import Batch
from src.neural_graphs.nn.inr import INR
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.autoencoder import create_batch_wb
from src.utils.helpers import set_seed

# For latent
from torch_geometric.utils import to_dense_adj
from src.neural_graphs.nn.gnn import to_pyg_batch



# ------------------------------
# Experiment funtionality
def inr_loss(
    ground_truth_image: torch.Tensor,
    reconstructed_image: torch.Tensor,
) -> torch.Tensor:
    """MSE loss function for INR."""
    return mse_loss(
        reconstructed_image,
        ground_truth_image,
        reduction="none",
    ).mean(dim=list(range(1, reconstructed_image.dim())))


def inr_loss_batches(
    batch: Batch | torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
    reconstructed: bool = False,
) -> torch.Tensor:
    """
    Compute the loss for a given dataset using the INR model.
    """
    
    weights_dev = [w.to(device) for w in batch.weights]
    biases_dev = [b.to(device) for b in batch.biases]
    imgs = test_inr(
        weights_dev,
        biases_dev,
        permuted_weights=reconstructed,
    )

    return loss_fn(imgs)


def compute_loss_matrix(
    interpolated_batches: list[Batch],
    mnist_ground_truth_img: torch.Tensor,
    device: torch.device,
    reconstructed: bool = False,
) -> torch.Tensor:
    """
    Compute the loss matrix for the given dataset using the INR model.

    Args:
        interpolated_batches (list[Batch]): List of batches to compute the loss for.
        mnist_ground_truth_img (torch.Tensor): Ground truth image for the dataset.
        device (torch.device): Device to perform the computation on.
        reconstructed (bool): Whether the weights are reconstructed or not.

    Returns:
        torch.Tensor: Loss matrix of shape (BATCH_SIZE, NUM_INTERPOLATION_SAMPLES).
    """
    loss_matrix = []
    for interpolated_batch in interpolated_batches:
        loss = inr_loss_batches(
            batch=interpolated_batch,
            loss_fn=partial(inr_loss, mnist_ground_truth_img),
            device=device,
            reconstructed=reconstructed,
        )
        loss_matrix.append(loss)
    loss_matrix = torch.stack(loss_matrix).permute(1, 0)
    return loss_matrix


@torch.no_grad()
def interpolation_experiment(
    cfg: DictConfig, 
    device: torch.device,
):
    net, loader = load_orbit_dataset_and_model(
        cfg=cfg,
        device=device,
    )
    
    perturbed_dataset_batches: list[Batch]
    perturbed_dataset_batches = perturb_inr_all_batches(
        loader=loader,
        perturbation = cfg.latent_analysis.perturbation,
        is_tuple_loader=False,
    )

    perturbed_dataset_inrs: list[INR]
    perturbed_dataset_inrs = instantiate_inr_all_batches(
        all_batches=perturbed_dataset_batches,
        device=device,
    )

    perturbed_dataset_inrs = perturbed_dataset_inrs[1:] + [perturbed_dataset_inrs[0]]

    loader_perturbed = create_tmp_torch_geometric_loader(
        dataset=perturbed_dataset_inrs,
        tmp_dir=cfg.latent_analysis.tmp_dir,
        cfg=cfg,
        device=device,
    )

    mnist_ground_truth_img = load_ground_truth_image(
        cfg.latent_analysis.mnist_ground_truth_img,
        device=device,
    )

    loss_matrix_original, loss_matrix_reconstruction, loss_matrix_latent = [], [], []
    for (wb_original), (wb_perturbed) in tqdm(
        zip(loader, loader_perturbed), desc="Processing batches", total=len(loader)
    ):
       
        # =====================================
        # 1.- Naive interpolation in weights space
        # =====================================
        wb_original = wb_original.to(device)
        inputs_original = (wb_original.weights, wb_original.biases)
      
        wb_perturbed = wb_perturbed.to(device)
        inputs_perturbed = (wb_perturbed.weights, wb_perturbed.biases)
  
        interpolated_batches: list[Batch]
        interpolated_batches = interpolate_batch(
            wb_original, wb_perturbed, NUM_INTERPOLATION_SAMPLES, 
        )
   
        loss_matrix = compute_loss_matrix(
            interpolated_batches=interpolated_batches,
            mnist_ground_truth_img=mnist_ground_truth_img,
            device=device,
            reconstructed=True,
        )
     
        loss_matrix_original.append(loss_matrix)
       
  
        # =====================================
        # 2.- Autoencoder interpolation in weights space
        # =====================================
        w_reconstructed_original, b_reconstructed_original = create_batch_wb(
            net(inputs_original),
        )
    
        w_reconstructed_perturbed, b_reconstructed_perturbed = create_batch_wb(
            net(inputs_perturbed),
        )

        wb_reconstructed_original = Batch(
            weights=w_reconstructed_original,
            biases=b_reconstructed_original,
            label=wb_original.label,
        )

        wb_reconstructed_perturbed = Batch(
            weights=w_reconstructed_perturbed,
            biases=b_reconstructed_perturbed,
            label=wb_perturbed.label,
        )
        interpolated_batches_reconstruction: list[Batch]
        interpolated_batches_reconstruction = interpolate_batch(
            wb_reconstructed_original,
            wb_reconstructed_perturbed,
            num_samples=NUM_INTERPOLATION_SAMPLES,
        )
        loss_matrix = compute_loss_matrix(
            interpolated_batches=interpolated_batches_reconstruction,
            mnist_ground_truth_img=mnist_ground_truth_img,
            device=device,
        
        )
        loss_matrix_reconstruction.append(loss_matrix)

        # =====================================
        # 3.- Interpolation in latent space (Neural Graphs)
        # =====================================

        def _ng_encode(model, wb_batch):
            """Encode a Batch (weights,biases) into z using the NG pipeline."""
            inputs = (wb_batch.weights, wb_batch.biases)
            node_feats, edge_feats, _ = model.construct_graph(inputs)
            num_nodes = node_feats.shape[1]

            pyg_b = to_pyg_batch(node_feats, edge_feats, model.edge_index)
            out_node, out_edge = model.gnn(
                x=pyg_b.x, edge_index=pyg_b.edge_index, edge_attr=pyg_b.edge_attr
            )

            node_feats = out_node.reshape(pyg_b.num_graphs, num_nodes, out_node.shape[-1])
            edge_feats = to_dense_adj(pyg_b.edge_index, pyg_b.batch, out_edge)
            z = model.pool(node_feats, edge_feats)   # [BATCH_SIZE, latent_dim]
            return z

        def _ng_decode(model, z):
            """Decode latent z back to flat WB prediction (compatible with create_batch_wb)."""
            decoded = model.proj_out(z)
            return decoded

        # Encode both endpoints
        latent_original  = _ng_encode(net, wb_original)   # [BATCH_SIZE, latent_dim]
        latent_perturbed = _ng_encode(net, wb_perturbed)  # [BATCH_SIZE, latent_dim]

        # Interpolate in latent space
        latents = interpolate_latent_batch(  # [NUM_INTERPOLATION_SAMPLES, BATCH_SIZE, latent_dim]
            latent_original,
            latent_perturbed,
            num_samples=NUM_INTERPOLATION_SAMPLES,
        )

        # Decode each intermediate latent to WB, then compute loss
        interpolated_batches_latent: list[Batch] = []
        for z_mid in latents:
            decoded = _ng_decode(net, z_mid)                 # flat vector(s)
            w_mid, b_mid = create_batch_wb(decoded)          # -> lists of tensors
            interpolated_batches_latent.append(
                Batch(weights=w_mid, biases=b_mid, label=wb_original.label)
            )

        loss_matrix = compute_loss_matrix(
            interpolated_batches=interpolated_batches_latent,
            mnist_ground_truth_img=mnist_ground_truth_img,
            device=device,
            reconstructed=False,   # decoded from latent (not permuted)
        )
        loss_matrix_latent.append(loss_matrix)

    loss_matrix_original = torch.cat(loss_matrix_original, dim=0)
    loss_matrix_reconstruction = torch.cat(loss_matrix_reconstruction, dim=0)
    loss_matrix_latent = torch.cat(loss_matrix_latent, dim=0)

    remove_tmp_torch_geometric_loader(
        tmp_dir=cfg.latent_analysis.tmp_dir,
    )

    return loss_matrix_original, loss_matrix_reconstruction, loss_matrix_latent

    #print("[DEBUG] Calling plot_interpolation_curves")
    #plot_interpolation_curves(
    #    loss_matrices=[
    #        (loss_matrix_original, "Original"),
    #        (loss_matrix_reconstruction, "Reconstructed")
    #    ],
    #    save_path=cfg.latent_analysis.image_save_path,
    #)
    #print("[DEBUG] Finished plot_interpolation_curves")
   

# ------------------------------
def get_args():
    p = argparse.ArgumentParser(add_help=False)   # note: disable default help so “-h” still works
    p.add_argument("--ckpt",    type=str, default="outputs/2025-05-11/16-21-51/5gzpb5lt/best_val.ckpt")
    p.add_argument("--split",   type=str, default="test", choices=["train","val","test"])
    p.add_argument("--outdir",  type=str, default="analysis/resources/interpolation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    p.add_argument(
        "--tmp_dir",
        type=str,
        default="analysis/tmp_dir",
    )

    p.add_argument(
        "--dataset_size",
        type=int,
        default=512,
        help="Number of augmented INRs to generate",
    )
    
    p.add_argument("--save_path", type=str, default="analysis/resources/interpolation/interpolation_ng.png")
    p.add_argument("--mnist_ground_truth_img", type=str, default="data/mnist/train/2/23089.png")

    p.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="Number of runs to perform",
    )
    p.add_argument(
        "--perturbation",
        type=float,
        default=0.005,
        help="Perturbation to apply to the INR weights",
    )

    p.add_argument(
        "--dataset_path",
        type=str,
        default="analysis/tmp_dir/orbit",
    )

    p.add_argument(
        "--split_path",
        type=str,
        default="analysis/tmp_dir/orbit/mnist_orbit_splits.json",
    )

    p.add_argument(
        "--save_matrices",
        action="store_true",
        help="Save the loss matrices to disk",
    )

    p.add_argument(
        "--orbit_transformation",
        type=str,
        default="PD",
        choices=["PD", "P", "D"],
        help="Type of transformation to apply to create the orbit dataset",
    )

    
    args, unknown = p.parse_known_args()
    # strip your args out of sys.argv so Hydra never sees them
    sys.argv = [sys.argv[0]] + unknown
    return args


# **must** be done before Hydra is ever imported
args = get_args()
import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="../src/neural_graphs/experiments/inr_classification/configs", config_name="base", version_base=None)
def main(cfg):
    """
    Run the interpolation experiment with the given arguments and configuration.
    """
    """
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run
    conf["batch_size"] = BATCH_SIZE
    """
    OmegaConf.set_struct(cfg, False)
    # build a tiny override config
    override = {
        "latent_analysis":   { "ckpt_path": args.ckpt, 
                              "split": args.split, "outdir": args.outdir, "seed": args.seed, 
                              "dataset_size": args.dataset_size, 
                              "debug": args.debug, "tmp_dir": args.tmp_dir,
                              "mnist_ground_truth_img": args.mnist_ground_truth_img,
                              "image_save_path": args.save_path,
                              "perturbation": args.perturbation,
                              "dataset_path": args.dataset_path,
                              "num_runs": args.num_runs,
                              "split_path": args.split_path,
                              "save_matrices": args.save_matrices,
                              "transform_type": args.orbit_transformation,  # or "sign_flipping"
                                },
            }
    
    # merge it into your main cfg (this will add any new keys under model/data)
    cfg = OmegaConf.merge(cfg, override)
    cfg.batch_size = BATCH_SIZE
    # now everything lives in cfg
   
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(cfg.latent_analysis.outdir, exist_ok=True)
    set_seed(cfg.latent_analysis.seed)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_runs = cfg.latent_analysis.num_runs

    # Sample random INRs from the test set
    with open(cfg.data.splits_path, 'r') as f:
        splits = json.load(f)
    sampled_inr_paths = random.sample(splits["test"]["path"], num_runs)
   
    #analysis/tmp_dir
    orbit_dataset_path = cfg.latent_analysis.tmp_dir + "/orbit"
    cfg.latent_analysis.tmp_dir = cfg.latent_analysis.tmp_dir + "/perturbed_orbit"  # Passed as argument to interpolation_experiment
    
    loss_matrix_original_list = []
    loss_matrix_reconstruction_list = []
    loss_matrix_latent_list = []

    for experiment_id, inr_path in enumerate(sampled_inr_paths):
        
        print("-" * 50)
        print(f"Running experiment {experiment_id + 1}/{num_runs}...")
        ## Create the orbit dataset of the INR
        generate_orbit_dataset(
            output_dir=orbit_dataset_path,
            inr_path=inr_path,
            device=device,
            dataset_size=cfg.latent_analysis.dataset_size,
            transform_type= cfg.latent_analysis.transform_type,
        )

        ## Run orbit interpolation experiment
        inr_label = inr_path.split("/")[-3].split("_")[-2]
        inr_id = inr_path.split("/")[-3].split("_")[-1]
        possible_pahts = [f"data/mnist/test/{inr_label}/{inr_id}.png", f"data/mnist/train/{inr_label}/{inr_id}.png"]

        # The image is either in the train or test set
        for path in possible_pahts:
            if os.path.exists(path):
                cfg.latent_analysis.mnist_ground_truth_img = path
            break
        else:
            raise FileNotFoundError(f"None of the paths exist: {possible_pahts}")

        # Sample INR to create orbit
        cfg.latent_analysis.image_save_path = f"analysis/resources/interpolation/interpolation_expid={experiment_id}_inrlabel={inr_label}.png"

        loss_matrix_original, loss_matrix_reconstruction, loss_matrix_latent = interpolation_experiment(
            cfg=cfg, 
            device=device,
        )

        # Clear unused variables and free memory
        delete_orbit_dataset(orbit_dataset_path)

        loss_matrix_original_list.append(loss_matrix_original)
        loss_matrix_reconstruction_list.append(loss_matrix_reconstruction)
        loss_matrix_latent_list.append(loss_matrix_latent)

        gc.collect()
        torch.cuda.empty_cache()
    
    # Concatenate the matrices after the loop
    loss_matrix_original_list = torch.cat(loss_matrix_original_list, dim=0)
    loss_matrix_reconstruction_list = torch.cat(loss_matrix_reconstruction_list, dim=0)
    loss_matrix_latent_list = torch.cat(loss_matrix_latent_list, dim=0)

    # Save the loss matrices
    output_dir = pathlib.Path("analysis/resources/interpolation/matrices")
    output_dir.mkdir(parents=True, exist_ok=True)

    
    # filename_original = f"loss_matrix-naiveng-{cfg.latent_analysis.transform_type}-numruns={num_runs}-perturbation={cfg.latent_analysis.perturbation}.pt"
    filename_reconstruction = f"loss_matrix-neural_graphs-{cfg.latent_analysis.transform_type}-numruns={num_runs}-perturbation={cfg.latent_analysis.perturbation}.pt"
    filename_latent = f"loss_matrix-latentng-{cfg.latent_analysis.transform_type}-numruns={num_runs}-perturbation={cfg.latent_analysis.perturbation}.pt"

    # torch.save(loss_matrix_original_list, output_dir / filename_original) # Not necessary to save naive if we obtain it from the other script 
    torch.save(loss_matrix_reconstruction_list, output_dir / filename_reconstruction)
    torch.save(loss_matrix_latent_list, output_dir / filename_latent)

    print(f"Saved loss matrices to {output_dir}")


if __name__ == "__main__":
    main()