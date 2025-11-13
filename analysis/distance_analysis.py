import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import torch_geometric
import yaml
from tqdm import tqdm

from src.utils.helpers import overwrite_conf, set_seed
from src.scalegmn.autoencoder import get_autoencoder
from src.data import dataset


def compute_distance_statistics(dist_matrix):
    """Compute statistics of the distance matrix.

    Args:
        dist_matrix: numpy array of shape [N, N] containing pairwise distances

    Returns:
        dict: Dictionary containing distance statistics
    """
    # Get upper triangle (excluding diagonal)
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    stats = {
        "mean": np.mean(upper_tri),
        "std": np.std(upper_tri),
        "min": np.min(upper_tri),
        "max": np.max(upper_tri),
        "median": np.median(upper_tri),
        "q25": np.percentile(upper_tri, 25),
        "q75": np.percentile(upper_tri, 75),
    }

    return stats


@torch.no_grad()
def collect_latents(model, loader, device):
    """Collects the latent codes from the model encoder for all samples in the dataset.

    Args:
        model: The model to use for encoding.
        loader: The data loader for the dataset.
        device: The device to use for computation.

    Returns:
        zs: The latent codes (tensor of shape [N, latent_dim]).
        ys: The labels (tensor of shape [N]).
        wbs: The raw INR parameters (list of tensors).
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


def analyze_distances(zs, outdir):
    """Analyze distances between latent vectors and generate visualizations.

    Args:
        zs: Tensor of shape [N, latent_dim] containing latent vectors
        outdir: Directory to save visualizations
    """
    # Compute euclidean distance matrix between the latent vectors
    dist_matrix = torch.cdist(zs, zs, p=2)
    dist_matrix = dist_matrix.cpu().numpy()

    # Compute and print distance statistics
    stats = compute_distance_statistics(dist_matrix)
    print("\nDistance Statistics:")
    print("-" * 40)
    for stat_name, value in stats.items():
        print(f"{stat_name}: {value:.4f}")

    # Plot the distance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(dist_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Euclidean Distance Matrix")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "euclidean_distance_matrix.png"), dpi=300)
    plt.close()
    print(
        "\nEuclidean distance matrix saved to:",
        os.path.join(outdir, "euclidean_distance_matrix.png"),
    )

    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    plt.hist(upper_tri, bins=50, density=True)
    plt.title("Distribution of Pairwise Distances")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "distance_distribution.png"), dpi=300)
    plt.close()
    print(
        "Distance distribution plot saved to:",
        os.path.join(outdir, "distance_distribution.png"),
    )


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
        default="data/mnist-inrs",
    )
    p.add_argument(
        "--split_path",
        type=str,
        default="data/mnist-inrs/mnist_splits.json",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="models/mnist_cls/scalegmn_autoencoder_baseline/scalegmn_autoencoder_baseline_mnist_rec.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument("--outdir", type=str, default="latent/resources/distance_analysis")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Initialize
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

    # Load the orbit dataset
    conf["data"]["dataset_path"] = args.dataset_path
    conf["data"]["split_path"] = args.split_path
    split_set = dataset(
        conf["data"],
        split="test",
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=True,
        get_first_layer_mask=True,
        return_wb=True,
    )
    loader = torch_geometric.loader.DataLoader(
        split_set, batch_size=conf["batch_size"], shuffle=False
    )
    print("Loaded", len(split_set), "samples in the dataset.")

    # Load the model
    decoder_hidden_dim_list = [
        conf["scalegmn_args"]["d_hid"] * elem
        for elem in conf["decoder_args"]["d_hidden"]
    ]
    conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list
    conf["scalegmn_args"]["layer_layout"] = split_set.get_layer_layout()

    net = get_autoencoder(conf, autoencoder_type="inr").to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    # Collect latent representations
    zs, _, _ = collect_latents(net, loader, device)

    # Analyze distances
    analyze_distances(zs, args.outdir)
