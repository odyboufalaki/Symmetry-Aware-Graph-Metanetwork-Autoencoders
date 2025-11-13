import torch
import argparse
import os

from src.utils.helpers import set_seed
from analysis.utils import collect_latents, load_orbit_dataset_and_model


"""
python latent/interpolation.py \
    --conf configs/mnist_rec/scalegmn_autoencoder_ablation.yml \
    --dataset_path data/mnist-inrs-orbit \
    --split_path data/mnist-inrs-orbit/mnist_orbit_splits.json \
    --ckpt models/mnist_rec_scale/scalegmn_autoencoder_ablation/scalegmn_autoencoder_baseline_mnist_rec_ablation.pt \
    --split test
"""

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
        default="data/mnist-inrs-orbit",
    )
    p.add_argument(
        "--split_path",
        type=str,
        default="data/mnist-inrs-orbit/mnist_orbit_splits.json",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="models/mnist_rec_scale/scalegmn_autoencoder/scalegmn_autoencoder_mnist_rec.pt",
        help="Path to model checkpoint (.pt or .ckpt)",
    )
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--outdir", type=str, default="latent/resources/orbit_analysis")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    torch.set_float32_matmul_precision("high")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net, loader = load_orbit_dataset_and_model(
        conf=args.conf,
        dataset_path=args.dataset_path,
        split_path=args.split_path,
        ckpt_path=args.ckpt,
        device=device,
    )

    zs, _, _ = collect_latents(net, loader, device)  # [DATASET_SIZE, latent_dim]

    ## Distance matrix
    # Compute euclidean distance matrix between the latent vectors
    dist_matrix = torch.cdist(zs, zs, p=2)
    dist_matrix = dist_matrix.cpu().numpy()

    # Plot the distance matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(dist_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Euclidean Distance Matrix")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "euclidean_distance_matrix.png"), dpi=300)
    plt.close()
    print("Euclidean distance matrix saved to:", os.path.join(args.outdir, "euclidean_distance_matrix.png"))

    ## UMAP of the orbit
    umap_reducer = DimensionalityReducer(
        method="umap",
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
    )
    
    umap_reducer.full_pipeline(
        model=net,
        loader=loader,
        device=device,
        save_path=os.path.join(args.outdir, "umap"),
    )

if __name__ == "__main__":
    main()