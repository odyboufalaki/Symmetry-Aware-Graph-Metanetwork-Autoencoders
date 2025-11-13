import torch
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.manifold import TSNE
import argparse
import os
import yaml
import torch_geometric
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange
from src.phase_canonicalization.test_inr import test_inr
from src.scalegmn.autoencoder import create_batch_wb
from src.neural_graphs.experiments.utils import count_parameters, ddp_setup, set_logger, set_seed
from torch_geometric.utils import to_dense_adj
from src.neural_graphs.nn.gnn import to_pyg_batch
import sys

def get_args():
    p = argparse.ArgumentParser(add_help=False)   # note: disable default help so “-h” still works
    p.add_argument("--ckpt",    type=str, default="outputs/2025-05-11/16-21-51/5gzpb5lt/best_val.ckpt")
    p.add_argument("--split",   type=str, default="test", choices=["train","val","test"])
    p.add_argument("--outdir",  type=str, default="analysis/resources/visualization")
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--pca_dim", type=int, default=None)
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


# --------------------------------------------------
def dimensionality_reduction(z, labels, method="umap", pca_dim=None, **kwargs):
    Z = z.numpy()
    if pca_dim is not None and pca_dim > 0 and Z.shape[1] > pca_dim:
        Z = PCA(n_components=pca_dim).fit_transform(Z)
    if method == "umap":
        reducer = umap.UMAP(**kwargs)
    elif method == "tsne":
        reducer = TSNE(**kwargs)
    else:
        raise ValueError(method)

    Y = reducer.fit_transform(Z)
    return Y

def scatter(Y, labels, title, out_png):
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    plt.gca().add_artist(legend1)
    plt.savefig(out_png, dpi=300)
    plt.close()


# --------------------------------------------------
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

    # load dataset
    if cfg.latent_analysis.split == "train":
        split_set = hydra.utils.instantiate(cfg.data.train)
    elif cfg.latent_analysis.split == "val":
        split_set = hydra.utils.instantiate(cfg.data.val)
    elif cfg.latent_analysis.split == "test":
        split_set = hydra.utils.instantiate(cfg.data.test)
    #plot_epoch_set = hydra.utils.instantiate(cfg.data.plot_epoch)
    

    loader = torch.utils.data.DataLoader(
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

    # ---------------- Load conf & build dataset ----------
    #conf = yaml.safe_load(open(args.conf))
    #conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #decoder_hidden_dim_list = [conf["scalegmn_args"]["d_hid"]*elem for elem in conf["decoder_args"]["d_hidden"]] 
    #conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  

    # ---------------- Model ------------------------------
    checkpoint = torch.load(cfg.latent_analysis.ckpt_path, map_location=device)
    # Extract the model's state_dict from the checkpoint
    model_state_dict = checkpoint['model']
    # Now load the extracted state_dict into your network
    model.load_state_dict(model_state_dict)
    model.eval()

    # ---------------- Collect latents --------------------
    global zs  # used inside decode_grid
    zs, ys = collect_latents(model, loader, device)
    print(
        f"Collected {len(zs)} latent codes of dim {zs.shape[1]} from {cfg.latent_analysis.split} split."
    )

    # ---------------- Dim-red ----------------------------

    emb_umap = dimensionality_reduction(
        zs,
        ys,
        method="umap",
        pca_dim=cfg.latent_analysis.pca_dim,
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
    )
    emb_tsne = dimensionality_reduction(
        zs,
        ys,
        method="tsne",
        pca_dim=cfg.latent_analysis.pca_dim,
        perplexity=10,
        init="pca",
        learning_rate="auto",
    )

    scatter(emb_umap, ys, f"UMAP - {cfg.latent_analysis.split}", os.path.join(cfg.latent_analysis.outdir, "umap_ng.png"))
    scatter(emb_tsne, ys, f"t-SNE - {cfg.latent_analysis.split}", os.path.join(cfg.latent_analysis.outdir, "tsne_ng.png"))
    print("Saved 2-D scatter plots to", cfg.latent_analysis.outdir)


"""
def train_ddp(rank, cfg, hydra_cfg):
    ddp_setup(rank, cfg.distributed.world_size)
    cfg.distributed.rank = rank
    train(cfg, hydra_cfg)
    destroy_process_group()
"""

# --------------------------------------------------


@hydra.main(config_path="../src/neural_graphs/experiments/inr_classification/configs", config_name="base", version_base=None)
def main(cfg):

  
    OmegaConf.set_struct(cfg, False)
    # build a tiny override config
    override = {
        "latent_analysis":   { "ckpt_path": args.ckpt, 
                              "split": args.split, "outdir": args.outdir, "seed": args.seed, "pca_dim": args.pca_dim }
   
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

