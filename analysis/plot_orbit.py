# ────────────────────────────────────────────────────────────
#  plot_orbit.py
#
#  Visualise the orbit of a single INR in the latent space of
#  an autoencoder and compare it with the global MNIST manifold.
# ────────────────────────────────────────────────────────────
import argparse
import copy
import os
import pathlib
import joblib
import yaml
import torch
import umap
import matplotlib.pyplot as plt
import torch_geometric
from tqdm import tqdm

from src.utils.helpers import overwrite_conf, set_seed
from src.scalegmn.autoencoder import get_autoencoder
from src.data import dataset

from utils import plotting as plt_utils
"""
python latent/plot_orbit.py \
    --conf configs/mnist_rec/scalegmn_autoencoder_ablation.yml \
    --ckpt models/mnist_rec_scale/scalegmn_autoencoder_ablation/scalegmn_autoencoder_baseline_mnist_rec_ablation.pt \
    --orbit_json data/mnist-inrs-orbit/mnist_orbit_splits.json \
    --fit_split test \
    --outdir latent/resources/orbit_plots
"""

# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────
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
    for batch, wb in loader:
        batch = batch.to(device)
        z = model.encoder(batch)  # [B, latent_dim]
        zs.append(z.cpu())
        ys.append(batch.label.cpu())
        wbs.append(wb)  # raw INR params, useful for reconstructions
    return torch.cat(zs), torch.cat(ys), wbs

def save_scatter(background, bg_labels, orbit_emb, title, out_png):
    """background : (N,2), orbit_emb : (M,2)"""
    plt.figure(figsize=(6, 6))
    
    colors_dict = {
        "0": ("Red", 400),
        "1": ("Orange", 400),
        "2": ("Yellow", 400),
        "3": ("Green", 400),
        "4": ("Cyan", 400),
        "5": ("Blue", 400),
        "6": ("Purple", 400),
        "7": ("Magenta", 400),
        "8": ("Red", 900),
        "9": ("Base", 600),
    }
    
    colors = [plt_utils.flexoki(*colors_dict[str(label)]) for label in bg_labels]
    
    # faint background MNIST manifold
    plt.scatter(
        background[:, 0],
        background[:, 1],
        c=colors,
        cmap="tab10",
        s=6,
        alpha=0.15,
        linewidths=0,
    )
    # orbit itself
    plt.scatter(
        orbit_emb[:, 0],
        orbit_emb[:, 1],
        c=plt_utils.flexoki("Yellow", 300),
        edgecolors=plt_utils.flexoki("Base", 950),
        s=120,
        alpha=1.0,
        marker="*",
        zorder=3,
        label="orbit pts",
        linewidths=0.5,
    )

    # optionally connect the orbit samples (helps show loops)
    plt.plot(orbit_emb[:, 0], orbit_emb[:, 1], lw=0.6, alpha=0.6, c=plt_utils.flexoki("Base", 800))
    plt.axis("off")
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("➜ saved plot →", out_png)


# ────────────────────────────────────────────────────────────
#  Arg-parsing
# ────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Visualise an INR orbit in latent space")
    p.add_argument(
        "--conf",
        type=str,
        default="configs/mnist_rec/scalegmn_autoencoder_ablation.yml",
        help="YAML config used at train",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        # default="models/mnist_rec_scale/scalegmn_autoencoder_ablation/scalegmn_autoencoder_baseline_mnist_rec_ablation.pt",
        default="models/mnist_rec_scale/scalegmn_autoencoder/scalegmn_autoencoder_mnist_rec.pt",
        help="Path to .pt model weights"
    )
    p.add_argument(
        "--orbit_json",
        type=str,
        default="data/mnist-inrs-orbit/mnist_orbit_splits.json",
        help="splits JSON written by orbit_dataset.py",
    )
    p.add_argument(
        "--fit_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="which split of the ORIGINAL data to fit UMAP on",
    )
    p.add_argument("--outdir", type=str, default="analysis/resources/orbit_plots")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n_neighbors",
        type=int,
        default=10,
        help="UMAP hyper-params (keep same as dim_red.py)",
    )
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    return p.parse_args()


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────
def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args = get_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ── load config & massage decoder dims exactly like dim_red.py ──
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, {"debug": False})
    decoder_hidden_dim_list = [
        conf["scalegmn_args"]["d_hid"] * elem
        for elem in conf["decoder_args"]["d_hidden"]
    ]
    conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list
    conf["scalegmn_args"]["layer_layout"] = [2, 32, 32, 1]

    # ── build & load model ──
    net = get_autoencoder(conf, autoencoder_type="inr").to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    # ── ORIGINAL latent cloud that we will *fit* UMAP on ───────────────────────
    orig_set = dataset(
        conf["data"],
        split=args.fit_split,
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=True,
        get_first_layer_mask=True,
        return_wb=True,
    )
    orig_loader = torch_geometric.loader.DataLoader(
        orig_set, batch_size=conf["batch_size"], shuffle=False
    )
    Z_train, y_train, _ = collect_latents(net, orig_loader, device)

    # ── fit UMAP ───────────────────────────────────────────────────────────────
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric=args.metric,
        random_state=args.seed,
    )
    reducer.fit(Z_train.numpy())
    joblib_path = pathlib.Path(args.outdir) / "umap_reducer.joblib"
    joblib.dump(reducer, joblib_path)
    print("✓ UMAP fitted on", Z_train.shape[0], "points  →", joblib_path)

    # ── ORBIT dataset loader (uses new splits json) ────────────────────────────
    orbit_conf = copy.deepcopy(conf)
    orbit_conf["data"]["split_path"] = args.orbit_json
    # force the dataset wrapper NOT to shuffle label order
    orbit_set = dataset(
        orbit_conf["data"],
        split="test",
        direction=conf["scalegmn_args"]["direction"],
        equiv_on_hidden=True,
        get_first_layer_mask=True,
        return_wb=True,
    )
    orbit_loader = torch_geometric.loader.DataLoader(
        orbit_set, batch_size=conf["batch_size"], shuffle=False
    )
    Z_orbit, _, _ = collect_latents(net, orbit_loader, device)
    print("✓ collected", Z_orbit.shape[0], "orbit samples")

    # ── embed both clouds ──────────────────────────────────────────────────────
    emb_train = reducer.transform(Z_train.numpy())
    emb_orbit = reducer.transform(Z_orbit.numpy())

    # ── plot ───────────────────────────────────────────────────────────────────
    base_name = (
        pathlib.Path(args.ckpt).stem.replace(".pt", "")
        + "_orbit"
        + f"_{args.fit_split}"
    )
    fig_path = pathlib.Path(args.outdir) / f"{base_name}.png"
    save_scatter(
        emb_train,
        y_train.numpy(),
        emb_orbit,
        title=base_name,
        out_png=str(fig_path),
    )


if __name__ == "__main__":
    main()
