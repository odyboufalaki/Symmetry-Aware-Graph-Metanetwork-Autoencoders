from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import torch
import argparse
import os
import logging
import numpy as np
import torch
from src.neural_graphs.experiments.utils import (
    count_parameters,
    set_seed,
)
from torch_geometric.utils import to_dense_adj
from src.neural_graphs.nn.gnn import to_pyg_batch
import sys


def get_args():
    p = argparse.ArgumentParser(
        add_help=False
    )  # note: disable default help so “-h” still works
    p.add_argument(
        "--ckpt", type=str, default="outputs/2025-05-11/16-21-51/5gzpb5lt/best_val.ckpt"
    )
    p.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    p.add_argument("--clusters", type=int, default=10, help="Number of KMeans clusters")
    p.add_argument("--seed", type=int, default=0)
    args, unknown = p.parse_known_args()
    # strip your args out of sys.argv so Hydra never sees them
    sys.argv = [sys.argv[0]] + unknown
    return args


# **must** be done before Hydra is ever imported
args = get_args()

import hydra
from omegaconf import OmegaConf


# --------------------------------------------------
def kmeans_classify(
    zs: np.ndarray, ys: np.ndarray, n_clusters: int = 10, random_state: int = 0
):
    """
    Fit KMeans on latent codes zs and map clusters to true labels by majority vote.
    Returns accuracy, predictions, cluster IDs, and mapping dict.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_ids = kmeans.fit_predict(zs)

    mapping = {}
    for c in range(n_clusters):  # produce mapping for each cluster
        mask = cluster_ids == c  # pick samples in cluster c
        if not mask.any():  # no samples in this cluster
            mapping[c] = -1
        else:
            mapping[c] = int(
                np.bincount(ys[mask]).argmax()
            )  # most common label in cluster c

    y_pred = np.array([mapping[c] for c in cluster_ids])  # map cluster IDs to labels
    acc = accuracy_score(ys, y_pred)  # compute accuracy
    return acc, y_pred, cluster_ids, mapping


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
def latents(cfg, hydra_cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        set_seed(cfg.seed)

    rank = OmegaConf.select(cfg, "distributed.rank", default=0)
    # ckpt_dir = Path(hydra_cfg.runtime.output_dir) / "checkpoints"

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
    # plot_epoch_set = hydra.utils.instantiate(cfg.data.plot_epoch)

    loader = torch.utils.data.DataLoader(
        dataset=split_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    if rank == 0:
        logging.info(f"dataset size {len(split_set)}, ")

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

    # os.makedirs(cfg.latent_analysis.outdir, exist_ok=True)
    set_seed(cfg.latent_analysis.seed)

    # ---------------- Load conf & build dataset ----------
    # conf = yaml.safe_load(open(args.conf))
    # conf = overwrite_conf(conf, {"debug": False})  # ensure standard run

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # decoder_hidden_dim_list = [conf["scalegmn_args"]["d_hid"]*elem for elem in conf["decoder_args"]["d_hidden"]]
    # conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Model ------------------------------
    checkpoint = torch.load(cfg.latent_analysis.ckpt_path, map_location=device)
    # Extract the model's state_dict from the checkpoint
    model_state_dict = checkpoint['model']
    # Now load the extracted state_dict into your network
    model.load_state_dict(model_state_dict)
    model.eval()

    # ---------------- Collect latents --------------------
    global zs, ys  # used inside decode_grid
    zs, ys = collect_latents(model, loader, device)
    print(
        f"Collected {len(zs)} latent codes of dim {zs.shape[1]} from {cfg.latent_analysis.split} split."
    )

    # ---------------- KMeans clustering ------------------
    zs = zs.cpu().numpy()
    ys = ys.cpu().numpy()
    acc, y_pred, cluster_ids, mapping = kmeans_classify(
        zs,
        ys,
        n_clusters=cfg.latent_analysis.clusters,
        random_state=cfg.latent_analysis.seed,
    )
    print(f"Accuracy: {acc:.4f}")
    print(f"Cluster IDs: {cluster_ids}")
    print(f"Mapping: {mapping}")
    # print(f"Cluster IDs: {cluster_ids}")
    # print(f"Mapping: {mapping}")


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):

    OmegaConf.set_struct(cfg, False)
    # build a tiny override config
    override = {
        "latent_analysis": {
            "ckpt_path": args.ckpt,
            "split": args.split,
            "seed": args.seed,
            "clusters": args.clusters,
        }
    }

    # merge it into your main cfg (this will add any new keys under model/data)
    cfg = OmegaConf.merge(cfg, override)
    # Override the Hydra cfg with those args
    # OmegaConf.update(cfg, "model.ckpt_path", args.ckpt)
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
    # else:
    latents(cfg, hydra_cfg)


if __name__ == "__main__":
    main()
