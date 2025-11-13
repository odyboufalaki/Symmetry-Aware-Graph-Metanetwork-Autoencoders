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
import sys
import torch_geometric
from tqdm import tqdm
import logging



from src.utils.helpers import overwrite_conf, set_seed
from src.scalegmn.autoencoder import get_autoencoder
from src.data import dataset

# --------------neural_graphs_imports----------------
from torch_geometric.utils import to_dense_adj
from src.neural_graphs.experiments.utils import count_parameters, set_logger, set_seed
from src.neural_graphs.nn.gnn import to_pyg_batch

from utils import plotting as plt_utils

"""
python latent/plot_orbit.py \
    --conf configs/mnist_rec/scalegmn_autoencoder_ablation.yml \
    --ckpt models/mnist_rec_scale/scalegmn_autoencoder_ablation/scalegmn_autoencoder_baseline_mnist_rec_ablation.pt \
    --orbit_json data/mnist-inrs-orbit/mnist_orbit_splits.json \
    --fit_split test \
    --outdir latent/resources/orbit_plots
"""
def get_args():
    p = argparse.ArgumentParser(add_help=False)   # note: disable default help so “-h” still works
    p.add_argument("--ckpt",    type=str, default="outputs/2025-05-11/16-21-51/5gzpb5lt/best_val.ckpt")
    p.add_argument("--split",   type=str, default="test", choices=["train","val","test"])
    p.add_argument("--outdir",  type=str, default="analysis/resources/visualization")
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument(
        "--n_neighbors",
        type=int,
        default=10,
    )
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    args, unknown = p.parse_known_args()
    # strip your args out of sys.argv so Hydra never sees them
    sys.argv = [sys.argv[0]] + unknown
    return args


# **must** be done before Hydra is ever imported
args = get_args()

import hydra
from omegaconf import OmegaConf

# --------------------------------------------------
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

# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────
def save_scatter(background, bg_labels, orbit_emb, title, out_png):
    """background : (N,2), orbit_emb : (M,2)"""
    plt.figure(figsize=(6, 6))
    # faint background MNIST manifold
    
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
#  Latents, main functionality
# ────────────────────────────────────────────────────────────
def latents(cfg, hydra_cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        set_seed(cfg.seed)

    rank = OmegaConf.select(cfg, "distributed.rank", default=0)
    #ckpt_dir = Path(hydra_cfg.runtime.output_dir) / "checkpoints"
    
    """
    if cfg.wandb.name is None:
        model_name = cfg.model._target_.split(".")[-1]
        cfg.wandb.name = (
            f"{cfg.data.dataset_name}_clf_{model_name}"
            f"_bs_{cfg.batch_size}_seed_{cfg.seed}"
        )
    """
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── ORIGINAL latent cloud that we will *fit* UMAP on ───────────────────────
    # load dataset
    if cfg.latent_analysis.split == "train":
        split_set = hydra.utils.instantiate(cfg.data.train)
    elif cfg.latent_analysis.split == "val":
        split_set = hydra.utils.instantiate(cfg.data.val)
    elif cfg.latent_analysis.split == "test":
        split_set = hydra.utils.instantiate(cfg.data.test)
    #plot_epoch_set = hydra.utils.instantiate(cfg.data.plot_epoch)
    

    orig_loader = torch.utils.data.DataLoader(
        dataset=split_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


    if rank == 0:
        logging.info(
            f"dataset size {len(split_set)}, "
        )

    point = split_set[0]
    weight_shapes = tuple(w.shape[:2] for w in point.weights)
    bias_shapes = tuple(b.shape[:1] for b in point.biases)

    layer_layout = [weight_shapes[0][0]] + [b[0] for b in bias_shapes]
    if rank == 0:
        logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")
    model_kwargs = dict()
    model_cls = cfg.model._target_.split(".")[-1]
    if model_cls == "DWSModelForClassification":
        model_kwargs["weight_shapes"] = weight_shapes
        model_kwargs["bias_shapes"] = bias_shapes
    else:
        model_kwargs["layer_layout"] = layer_layout
    model = hydra.utils.instantiate(cfg.model, **model_kwargs).to(device)

    if rank == 0:
        logging.info(f"number of parameters: {count_parameters(model)}")

    if cfg.compile:
        model = torch.compile(model, **cfg.compile_kwargs)
    

    os.makedirs(cfg.latent_analysis.outdir, exist_ok=True)
    set_seed(cfg.latent_analysis.seed)


    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


    # ---------------- Model ------------------------------
    checkpoint = torch.load(cfg.latent_analysis.ckpt_path, map_location=device)
    # Extract the model's state_dict from the checkpoint
    model_state_dict = checkpoint['model']
    # Now load the extracted state_dict into your network
    model.load_state_dict(model_state_dict)
    model.eval()
    
    Z_train, y_train = collect_latents(model, orig_loader, device)

    # ── fit UMAP ───────────────────────────────────────────────────────────────
    reducer = umap.UMAP(
        n_neighbors=cfg.latent_analysis.n_neighbors,
        min_dist=cfg.latent_analysis.min_dist,
        n_components=2,
        metric=cfg.latent_analysis.metric,
        random_state=cfg.latent_analysis.seed,
    )
    reducer.fit(Z_train.numpy())
    joblib_path = pathlib.Path(cfg.latent_analysis.outdir) / "umap_reducer.joblib"
    joblib.dump(reducer, joblib_path)
    print("✓ UMAP fitted on", Z_train.shape[0], "points  →", joblib_path)

    # ── ORBIT dataset loader (uses new splits json) ────────────────────────────

    # force the dataset wrapper NOT to shuffle label order
    orbit_set = hydra.utils.instantiate(cfg.data.orbit_plot)
  
    orbit_loader = torch.utils.data.DataLoader(
        dataset=orbit_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    Z_orbit, _ = collect_latents(model, orbit_loader, device)
    print("✓ collected", Z_orbit.shape[0], "orbit samples")

    # ── embed both clouds ──────────────────────────────────────────────────────
    emb_train = reducer.transform(Z_train.numpy())
    emb_orbit = reducer.transform(Z_orbit.numpy())

    # ── plot ───────────────────────────────────────────────────────────────────
    base_name = (
        "neural_graphs_"
        + "_orbit"
        + f"_{cfg.latent_analysis.split}"
    )
    fig_path = pathlib.Path(cfg.latent_analysis.outdir) / f"{base_name}.png"
    
    save_scatter(
        emb_train,
        y_train.numpy(),
        emb_orbit,
        title=base_name,
        out_png=str(fig_path),
    )

@hydra.main(config_path="../src/neural_graphs/experiments/inr_classification/configs", config_name="base", version_base=None)
def main(cfg):

  
    OmegaConf.set_struct(cfg, False)
    # build a tiny override config
    override = {
        "latent_analysis":   { "ckpt_path": args.ckpt, 
                              "split": args.split, "outdir": args.outdir, "seed": args.seed, 
                              "n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "metric": args.metric }
   
    }
    
    # merge it into your main cfg (this will add any new keys under model/data)
    cfg = OmegaConf.merge(cfg, override)
    # Override the Hydra cfg with those args
   #OmegaConf.update(cfg, "model.ckpt_path", args.ckpt)
   # OmegaConf.update(cfg, "data.split",       args.split)
   # OmegaConf.update(cfg, "data.outdir",      args.outdir)
   # OmegaConf.update(cfg, "seed",             args.seed)
   # OmegaConf.update(cfg, "pca_dim",          args.pca_dim)

    # now everything lives in cfg
   
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    """
    if cfg.distributed:
        mp.spawn(
            train_ddp,
            args=(cfg, hydra_cfg),
            nprocs=cfg.distributed.world_size,
            join=True,
        )
    """
    #else:
    latents(cfg, hydra_cfg)


if __name__ == "__main__":
    main()
