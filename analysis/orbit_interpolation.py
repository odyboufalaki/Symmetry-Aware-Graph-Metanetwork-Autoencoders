"""
Orbit interpolation experiment for the ScaleGNN model and the linear assignment.

A run of the interpolation experiment consists of:
1. Randomly choosing an INR from the test set.
2. Creating an orbit dataset of the INR.
3. Perturbing the INR parameters with a gaussian distributed noise (plus a perturbation parameter).
4. Performing interpolation in weights space:
    4.1 Naive interpolation: Between the original orbit INRs and the group-acted and perturbed INRs.
    4.2 Autoencoder interpolation: Between the reconstructed INRs of the original orbit INRs and the reconstructed INRs of the group-acted and perturbed INRs.
    4.3 Linear assignment interpolation: Between the original orbit INRs and the group-acted and perturbed INRs.
    4.4 Latent space interpolation: Between the latent representations
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
from functools import partial
from typing import Callable
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np

import torch
import yaml
from torch.nn.functional import mse_loss
from tqdm import tqdm

from analysis.linear_assignment import match_weights_biases_batch
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
    convert_and_prepare_weights,
    interpolate_latent_batch,
    NUM_INTERPOLATION_SAMPLES,
    BATCH_SIZE,
)
from analysis.utils.utils_sgmn import (
    create_tmp_torch_geometric_loader,
    load_orbit_dataset_and_model,
)
from analysis.utils import plotting as plt_utils
from src.data.base_datasets import Batch
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.autoencoder import create_batch_wb
from src.scalegmn.inr import INR
from src.utils.helpers import overwrite_conf, set_seed

# ------------------------------
# Argument parsing
# ------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--conf",
        type=str,
        default="configs/mnist_rec/scalegmn_autoencoder.yml",
        help="YAML config used during training",
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
        "--ckpt",
        type=str,
        default="models/mnist_rec_scale/scalegmn_autoencoder/scalegmn_autoencoder_mnist_rec.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument(
        "--tmp_dir",
        type=str,
        default="analysis/tmp_dir",
    )
    p.add_argument(
        "--dataset_size",
        type=int,
        default=512,  # 512
        help="Number of augmented INRs to generate",
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num_runs",
        type=int,
        default=10,  # 10
        help="Number of runs to perform",
    )
    p.add_argument(
        "--perturbation",
        type=float,
        default=0.005,
        help="Perturbation to apply to the INR weights",
    )
    p.add_argument(
        "--linear_assignment",
        type=str,
        default="PD",
        choices=["PD", "DP", "P", "D"],
        help="Type of linear assignment to use for matching weights and biases (PD, DP, P, D)",
    )
    p.add_argument(
        "--save_matrices",
        default=True,
        action="store_true",
        help="Save the loss matrices for later analysis",
    )
    p.add_argument(
        "--orbit_transformation",
        type=str,
        default="PD",
        choices=["PD", "P", "D"],
        help="Type of transformation to apply to create the orbit dataset",
    )
    p.add_argument(
        "--experiments",
        type=list[str],
        default=["naive", "scalegmn", "lap", "latent"],
        choices=["naive", "scalegmn", "lap", "latent"],
    )
    # ---- Plotting options ----
    p.add_argument(
        "--plot",
        default=True,
        action="store_true",
        help="Plot interpolation curves at the end."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="analysis/resources/interpolation",
        help="Directory to save plots (and matrices if --save_matrices).",
    )
    p.add_argument(
        "--band",
        type=str,
        default="std",
        choices=["sem", "std", "none"],
        help="Uncertainty band: standard error (sem), standard deviation (std), or none.",
    )
    p.add_argument(
        "--no_legend",
        default=True,
        action="store_true",
        help="Disable legend in the plot.",
    )
    return p.parse_args()


# ------------------------------
# Experiment funtionality
# ------------------------------
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


# ------------------------------
# Plotting helpers
# -------------------------------


def _curve_stats(matrix: torch.Tensor):
    """
    matrix: [N, NUM_INTERPOLATION_SAMPLES]
    returns mean, std, sem (numpy arrays) and N
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix [N, T], got shape {tuple(matrix.shape)}")

    mean = matrix.mean(dim=0).detach().cpu().numpy()
    std  = matrix.std(dim=0, unbiased=False).detach().cpu().numpy()
    n    = max(int(matrix.size(0)), 1)
    sem  = std / np.sqrt(n)
    return mean, std, sem, n


def plot_interpolation_curves_inline(
    curves: list[tuple[torch.Tensor, str]],
    title: str,
    save_path: str,
    with_legend: bool = True,
    band: str = "sem",  # "sem" | "std" | "none"
    colors_by_label: dict[str, str | tuple[str, int]] | None = None,  
):
    """
    curves: list of (loss_matrix, label), each loss_matrix is [N, NUM_INTERPOLATION_SAMPLES]
    """
    # X axis in [0,1]
    xs = np.linspace(0.0, 1.0, NUM_INTERPOLATION_SAMPLES)

    plt.figure(figsize=(7.0, 4.5), dpi=150)
    for matrix, label in curves:
        mean, std, sem, _ = _curve_stats(matrix)

        # Resolve color (hex or (name, shade) -> hex), else fall back to cycle
        color = None
        if colors_by_label and label in colors_by_label:
            spec = colors_by_label[label]
            color = plt_utils.flexoki(*spec) if isinstance(spec, tuple) else spec

        plt.plot(xs, mean, linewidth=2.0, label=label, color=color)

        if band != "none":
            band_vals = sem if band == "sem" else std
            lower = mean - band_vals
            upper = mean + band_vals
            plt.fill_between(xs, lower, upper, alpha=0.30, linewidth=0, color=color)

    plt.xlabel("Interpolation $t$")
    plt.ylabel("MSE loss")
    plt.title(title)
    if with_legend:
        plt.legend(frameon=False)
    plt.tight_layout()

    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()



@torch.no_grad()
def interpolation_experiment(
    args: argparse.Namespace,
    conf: dict, 
    device: torch.device,
):
    # Get the Autoencoder model and the orbit dataset loader
    net, loader = load_orbit_dataset_and_model(
        conf=conf,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ckpt_path=args.ckpt,
        device=device,
    )

    # Perturb the dataset batches
    # Perturbation is a gaussian noise added to the INR weights
    perturbed_dataset_batches: list[Batch]
    perturbed_dataset_batches = perturb_inr_all_batches(
        loader=loader,
        perturbation=args.perturbation,
    )
    perturbed_dataset_inrs: list[INR]
    perturbed_dataset_inrs = instantiate_inr_all_batches(
        all_batches=perturbed_dataset_batches,
        device=device,
    )
    perturbed_dataset_inrs = perturbed_dataset_inrs[1:] + [perturbed_dataset_inrs[0]]

    # Create torch gometric loader
    loader_perturbed = create_tmp_torch_geometric_loader(
        dataset=perturbed_dataset_inrs,
        tmp_dir=args.tmp_dir,
        conf=conf,
        device=device,
    )

    # Load ground truth image
    mnist_ground_truth_img = load_ground_truth_image(
        args.mnist_ground_truth_img,
        device=device,
    )

    #[BATCH_SIZE, NUM_INTERPOLATION_SAMPLES]
    loss_matrices = {
        key: [] for key in args.experiments
    }
    for (batch_original, wb_original), (batch_perturbed, wb_perturbed) in tqdm(
        zip(loader, loader_perturbed), desc="Processing batches", total=len(loader)
    ):
        batch_original = batch_original.to(device)
        batch_perturbed = batch_perturbed.to(device)

        wb_original = wb_original.to(device)
        wb_perturbed = wb_perturbed.to(device)

        #======================================================
        # 1. Naive interpolation in original weight space
        #======================================================
        
        if "naive" in args.experiments:
            # Interpolation in original weight space
            interpolated_batches: list[Batch]  # [NUM_INTERPOLATION_SAMPLES, Batch]
            interpolated_batches = interpolate_batch(
                wb_original, wb_perturbed, NUM_INTERPOLATION_SAMPLES, 
            )

            loss_matrices["naive"].append(
                compute_loss_matrix(
                    interpolated_batches=interpolated_batches,
                    mnist_ground_truth_img=mnist_ground_truth_img,
                    device=device,
                    reconstructed=True,
                )
            )

        #======================================================
        # 2. Interpolation in reconstructed weight space (ScaleGMN Encoder)
        #======================================================

        # Interpolation in reconstructed weight space
        if "scalegmn" in args.experiments:
            w_reconstructed_original, b_reconstructed_original = create_batch_wb(
                net(batch_original.clone()),
            )
            
            w_reconstructed_perturbed, b_reconstructed_perturbed = create_batch_wb(
                net(batch_perturbed.clone()),
            )
            wb_reconstructed_original = Batch(  # BATCH_SIZE, 2
                weights=w_reconstructed_original,
                biases=b_reconstructed_original,
                label=wb_original.label,
            )
            wb_reconstructed_perturbed = Batch(  # BATCH_SIZE, 2
                weights=w_reconstructed_perturbed,
                biases=b_reconstructed_perturbed,
                label=wb_perturbed.label,
            )

            interpolated_batches_reconstruction: list[Batch]  # [NUM_INTERPOLATION_SAMPLES, Batch]
            interpolated_batches_reconstruction = interpolate_batch(
                wb_reconstructed_original,
                wb_reconstructed_perturbed,
                num_samples=NUM_INTERPOLATION_SAMPLES,
            )

            loss_matrices["scalegmn"].append(
                compute_loss_matrix(
                    interpolated_batches=interpolated_batches_reconstruction,
                    mnist_ground_truth_img=mnist_ground_truth_img,
                    device=device,
                    reconstructed=False,  # Not reconstructed in this case
                )
            )

        #======================================================
        # 3. Linear assignment interpolation
        #======================================================
        if "lap" in args.experiments:
            # Match the weights and biases of the original and perturbed batches
            rebased_weights, rebased_biases = match_weights_biases_batch(
                weights_A_batch=wb_original.weights,
                weights_B_batch=wb_perturbed.weights,
                biases_A_batch=wb_original.biases,
                biases_B_batch=wb_perturbed.biases,
                matching_type=args.linear_assignment 
            )
            rebased_weights, rebased_biases = convert_and_prepare_weights(
                rebased_weights, rebased_biases, device=device
            )
            
            # rebased weights and biases
            wb_rebased = Batch(
                weights=rebased_weights,
                biases=rebased_biases,
                label=wb_perturbed.label,
            )
            
            interpolated_batches_transformation: list[Batch]  # [NUM_INTERPOLATION_SAMPLES, Batch]
            interpolated_batches_transformation = interpolate_batch(
                wb_original,
                wb_rebased,
                num_samples=NUM_INTERPOLATION_SAMPLES,
            )

            loss_matrices["lap"].append(
                compute_loss_matrix(
                    interpolated_batches=interpolated_batches_transformation,
                    mnist_ground_truth_img=mnist_ground_truth_img,
                    device=device,
                    reconstructed=True,
                )
            )

        #=======================================================
        # 4. Interpolation in latent space (ScaleGMN Encoder)
        #=======================================================
        if "latent" in args.experiments:
            latent_original = net.encoder(batch_original.clone())  # [BATCH_SIZE, 2]
            latent_perturbed = net.encoder(batch_perturbed.clone())  # [BATCH_SIZE, 2]

            # Interpolate in latent space
            latents = interpolate_latent_batch(  # [NUM_INTERPOLATION_SAMPLES, BATCH_SIZE, 2]
                latent_original,
                latent_perturbed,
                num_samples=NUM_INTERPOLATION_SAMPLES,
            )

            interpolated_batches_latent : list[Batch] = []  # [NUM_INTERPOLATION_SAMPLES, Batch]
            for intermediate_latent in latents:
                # Decode the intermediate latent
                w_reconstructed_original, b_reconstructed_original = create_batch_wb(
                    net.decoder(intermediate_latent),  # [BATCH_SIZE, 2]
                )
                wb_reconstructed_original = Batch(
                    weights=w_reconstructed_original,
                    biases=b_reconstructed_original,
                    label=wb_original.label,
                )
                interpolated_batches_latent.append(wb_reconstructed_original)

            loss_matrices["latent"].append(
                compute_loss_matrix(
                    interpolated_batches=interpolated_batches_latent,
                    mnist_ground_truth_img=mnist_ground_truth_img,
                    device=device,
                    reconstructed=False,  # Not reconstructed in this case
                )
            )
    # Convert to tensor
    loss_matrices = {key: torch.cat(loss_matrices[key], dim=0) for key in loss_matrices}

    # Delete tmp dir
    remove_tmp_torch_geometric_loader(
        tmp_dir=args.tmp_dir,
    )

    return loss_matrices


def main():
    """
    Run the main function with different orbits from different INRs.
    This function creates a new dataset with different orbits and runs the interpolation experiment.
    """
    args = get_args()
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run
    conf["batch_size"] = BATCH_SIZE

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    num_runs = args.num_runs

    # Sample random INRs from the test set
    splits = json.load(open(conf["data"]["split_path"]))
    sampled_inr_paths = random.sample(splits["test"]["path"], num_runs)
   
    orbit_dataset_path = args.tmp_dir + "/orbit"
    args.tmp_dir = args.tmp_dir + "/perturbed_orbit"  # Passed as argument to interpolation_experiment
    
    aggregated_loss_matrices = {
        key: [] for key in args.experiments
    }
    for experiment_id, inr_path in enumerate(sampled_inr_paths):
        conf = yaml.safe_load(open(args.conf))
        conf = overwrite_conf(conf, {"debug": False})  # ensure standard run
        conf["batch_size"] = BATCH_SIZE
        
        print("-" * 50)
        print(f"Running experiment {experiment_id + 1}/{num_runs}...")
        ## Create the orbit dataset of the INR
        generate_orbit_dataset(
            output_dir=orbit_dataset_path,
            inr_path=inr_path,
            device=device,
            dataset_size=args.dataset_size,
            transform_type=args.orbit_transformation,
        )

        ## Run orbit interpolation experiment
        inr_label = inr_path.split("/")[-3].split("_")[-2]
        inr_id = inr_path.split("/")[-3].split("_")[-1]
        possible_paths = [f"data/mnist/test/{inr_label}/{inr_id}.png", f"data/mnist/train/{inr_label}/{inr_id}.png"]

        # The image is either in the train or test set
        for path in possible_paths:
            if os.path.exists(path):
                args.mnist_ground_truth_img = path
                break
        else:
            raise FileNotFoundError(f"None of the paths exist: {possible_paths}")


      
        # if args.linear_assignment:
        #     loss_matrix_original, loss_matrix_reconstruction = linear_assignment_experiment(
        #         args=args,
        #         conf=conf, 
        #         device=device,
        #         matching_type=args.linear_assignment,
        #     )
        # else:
        #     loss_matrix_original, loss_matrix_reconstruction = interpolation_experiment(
        #         args=args,
        #         conf=conf, 
        #         device=device,
        #     )

        loss_matrices = interpolation_experiment(
            args=args,
            conf=conf, 
            device=device,
        )
        
        # Clear unused variables and free memory
        delete_orbit_dataset(orbit_dataset_path)

        for key in loss_matrices:
            if key in args.experiments:
                aggregated_loss_matrices[key].append(loss_matrices[key])

        gc.collect()
        torch.cuda.empty_cache()
    
    for key in aggregated_loss_matrices:
        aggregated_loss_matrices[key] = torch.cat(aggregated_loss_matrices[key], dim=0)

    # Save loss matrices for later analysis
    output_dir = "analysis/resources/interpolation/matrices"

    for key in aggregated_loss_matrices:
        loss_matrix_name = f"loss_matrix-{key}-{args.orbit_transformation}-numruns={num_runs}-perturbation={args.perturbation}.pt"
        torch.save(
            aggregated_loss_matrices[key],
            os.path.join(output_dir, loss_matrix_name),
        )
        print(f"[info] Saved {loss_matrix_name} in {output_dir}")
        
    print("Saved loss matrices")

    exit() # Should not be plotting
    # ------------------------------
    # Plot curves comparing experiments
    # ------------------------------
    if args.plot:
        label_map = {
            "naive": "Naive (weights)",
            "scalegmn": "Autoencoder (ScaleGMN)",
            "lap": "Linear Assignment (Re-basin)",
            "latent": "Latent space",
        }

        curves_to_plot = []
        for key in args.experiments:
            mat = aggregated_loss_matrices.get(key, None)
            if mat is None or mat.numel() == 0:
                continue
            curves_to_plot.append((mat, label_map.get(key, key)))

        if len(curves_to_plot) == 0:
            print("Nothing to plot (no matrices found for selected experiments).")
        else:
            out_png = (
                pathlib.Path(args.output_dir) /
                f"{args.orbit_transformation}_numruns={num_runs}_perturbation={args.perturbation}.png"
            )
            with_legend = (args.orbit_transformation == "PD") and (not args.no_legend)

            # Right before plot_interpolation_curves_inline(...)
            plt_utils.set_flexoki_cycle()  # global cycle (still useful for any unassigned colors)

            method_colors = {
                "Naive (weights)": ("Blue", 600),
                "Autoencoder (ScaleGMN)": ("Orange", 400),
                "Linear Assignment (Re-basin)": ("Purple", 600),
                "Latent space": ("Green", 500),
            }

            plot_interpolation_curves_inline(
                curves=curves_to_plot,
                title=f"Interpolation ({args.orbit_transformation}) — runs={num_runs}, perturb={args.perturbation}",
                save_path=str(out_png),
                with_legend=with_legend,
                band=args.band,
                colors_by_label=method_colors,  
            )
            print(f"Saved plot to {out_png}")



if __name__ == "__main__":
    main()