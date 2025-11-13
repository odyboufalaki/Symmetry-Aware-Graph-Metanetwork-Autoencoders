import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import trange, tqdm
import wandb
import hydra
from omegaconf import OmegaConf

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch_geometric
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
from torch.utils.data import Subset
from src.neural_graphs.experiments.utils import (
    count_parameters,
    set_logger,
    set_seed,
)
from src.neural_graphs.experiments.inr_classification.main import ddp_setup
from src.neural_graphs.nn.nfn.common.data import WeightSpaceFeatures, network_spec_from_wsfeat

from src.scalegmn.cnn import unflatten_params_batch, get_logits_for_batch
from src.scalegmn.utils import get_cifar10_test_loader

set_logger()

warnings.filterwarnings("ignore", ".*TypedStorage is deprecated.*")
OmegaConf.register_new_resolver("prod", lambda x, y: x * y)

@torch.no_grad()
def evaluate(
    net, 
    loader,
    criterion,
    cifar10_test_loader,
    device,
    layout,
    param_shapes,
    effective_conf
):
    """
    Evaluates the autoencoder by reconstructing models and measuring their performance.

    This function calculates:
    1. The KL-divergence loss between original and reconstructed model logits.
    2. The actual accuracy achieved by the reconstructed models.
    3. Regression metrics (R^2, Kendall's Tau) comparing the ground-truth
       accuracies with the reconstructed models' accuracies.
    """
    net.eval()
    
    # Lists to store results from each batch of models
    all_avg_losses = []
    all_recon_accuracies = []
    # TODO: Remember to remove
    all_actual_accuracies = []  # For actual accuracies of original models

    all_teacher_accuracies = []

    # Loop over the synchronized dataloaders for the set (e.g., validation set)
    #for graph_batch, original_params_batch in tqdm(zip(graph_loader, original_params_loader), desc="Evaluating", leave=False, total=len(graph_loader)):
    for (graph_batch, original_params_batch) in tqdm(loader, desc="Evaluating", leave=False, total=len(loader)):
        # Move data to device
        graph_batch = graph_batch.to(device)
        original_params_batch = original_params_batch.to(device)
        # Store the ground-truth (teacher) accuracies for this batch
        all_teacher_accuracies.append(original_params_batch.y.cpu().numpy())

        # 1. Reconstruct parameters from the graph representations
        recon_params_flat = net(graph_batch)
        recon_params = unflatten_params_batch(recon_params_flat, layout, param_shapes)
        original_params = (original_params_batch.weights, original_params_batch.biases)
        
        # --- Start evaluation on the full CIFAR-10 test set ---
        total_kl_div_loss = 0.0
        num_test_samples = 0
        
        # We also need to calculate the accuracy of the reconstructed models
        B_models = recon_params[0][0].shape[0]
        recon_correct_preds = torch.zeros(B_models, device=device)
        # TODO: Remember to remove
        actual_correct_preds = torch.zeros(B_models, device=device)  # For actual accuracies of original models
        total_elements = 0
        for cifar_images, cifar_labels in cifar10_test_loader:
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)

            # Get raw logits for both batches of models
            original_logits = get_logits_for_batch(original_params, cifar_images, effective_conf)
            recon_logits = get_logits_for_batch(recon_params, cifar_images, effective_conf)
            
            # --- Calculate Loss ---
            B_models, B_images, N_classes = recon_logits.shape
            num_test_samples += B_images

            recon_logits_flat = recon_logits.view(-1, N_classes)
            original_logits_flat = original_logits.view(-1, N_classes)
            current_elements = B_models * B_images
            batch_loss = criterion(recon_logits_flat, original_logits_flat) * current_elements
            total_elements += current_elements
            total_kl_div_loss += batch_loss.item() # Use .item() as we don't need the graph

            # --- Calculate Accuracy of Reconstructed Models ---
            # Get the predicted class for each model on each image
            _, predicted_class = torch.max(recon_logits, dim=2) # Shape: (B_models, B_images)

            # TODO: Remember to remove
            _, actual_class = torch.max(original_logits, dim=2) # Shape: (B_models, B_images)
            
            # Compare predictions to true labels (broadcasting handles the dimensions)
            # Shape of comparison: (B_models, B_images)
            correct_mask = (predicted_class == cifar_labels.unsqueeze(0))

            # TODO: Remember to remove
            actual_mask = (actual_class == cifar_labels.unsqueeze(0))
            # Sum correct predictions for each model in the batch and accumulate
            recon_correct_preds += correct_mask.sum(dim=1)

            # TODO: Remember to remove
            actual_correct_preds += actual_mask.sum(dim=1)
         

        # --- Finalize metrics for this batch of models ---
        # Average loss for this batch
        avg_loss_for_batch = total_kl_div_loss / (total_elements)
        all_avg_losses.append(avg_loss_for_batch)

        # Final accuracy for each reconstructed model in the batch
        batch_recon_accuracies = (recon_correct_preds / num_test_samples).cpu().numpy()
        all_recon_accuracies.append(batch_recon_accuracies)

        # TODO: Remember to remove
        batch_actual_accuracies = (actual_correct_preds / num_test_samples).cpu().numpy()
        all_actual_accuracies.append(batch_actual_accuracies)

    # --- Aggregate results from all batches ---
    # Concatenate results into single numpy arrays
    pred = np.concatenate(all_recon_accuracies)
    actual = np.concatenate(all_teacher_accuracies)
    #TODO: Remember to remove
    actual_from_logits = np.concatenate(all_actual_accuracies)  # For actual accuracies of original models
    # print the pred and actual values for comparison in debugging
    #print(f"Actual accuracies from logits: {actual_from_logits}")
    # Calculate final metrics
    avg_loss = np.mean(all_avg_losses)
    avg_err = np.mean(np.abs(pred - actual)) # L1 error between accuracies
    rsq = r2_score(actual, pred)
    tau = kendalltau(actual, pred).correlation

    return {
        "avg_loss": avg_loss,
        "avg_err": avg_err,
        "rsq": rsq,
        "tau": tau,
        "pred": pred, # The accuracies of the reconstructed models
        "actual": actual, # The accuracies of the original models
    }




def train(cfg, hydra_cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    OmegaConf.set_struct(cfg, False)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    rank = OmegaConf.select(cfg, "distributed.rank", default=0)
    ckpt_dir = Path(hydra_cfg.runtime.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.name is None:
        model_name = cfg.model._target_.split(".")[-1]
        cfg.wandb.name = f"cnn_gen_{model_name}" f"_bs_{cfg.batch_size}_seed_{cfg.seed}"
    if rank == 0:
        wandb.init(
            **OmegaConf.to_container(cfg.wandb, resolve=True),
            settings=wandb.Settings(start_method="fork"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # load dataset
    train_set = hydra.utils.instantiate(cfg.data.train)
    val_set = hydra.utils.instantiate(cfg.data.val)
    test_set = hydra.utils.instantiate(cfg.data.test)

    if "debug_subset_fraction" in cfg.data and cfg.data.debug_subset_fraction < 1.0:
        fraction = cfg.data.debug_subset_fraction
        print(f"!!! SWEEP MODE: Using a {fraction*100:.0f}% subset of the data !!!")

        # Create random indices for the subset
        train_size = len(train_set)
        train_indices = np.random.choice(range(train_size), size=int(train_size * fraction), replace=False)

        val_size = len(val_set)
        val_indices = np.random.choice(range(val_size), size=int(val_size * fraction), replace=False)

        test_size = len(test_set)
        test_indices = np.random.choice(range(test_size), size=int(test_size * fraction), replace=False)

        # Overwrite the dataset variables with the new Subset objects
        train_set = Subset(train_set, train_indices)
        val_set = Subset(val_set, val_indices)
        test_set = Subset(test_set, test_indices)

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=cfg.batch_size,
        shuffle=not cfg.distributed,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(train_set) if cfg.distributed else None,
    )
    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    cifar10_test_loader = get_cifar10_test_loader(
        batch_size=cfg.batch_size,
        num_workers= cfg.num_workers,
    )



    if rank == 0:
        logging.info(
            f"train size {len(train_set)}, "
            f"val size {len(val_set)}, "
            f"test size {len(test_set)}"
        )

    original_set = train_set.dataset if isinstance(train_set, Subset) else train_set

    param_shapes = {}
    for _, row in original_set.layout.iterrows():
        varname = row['varname']
        shape_str = row['shape']
        
        # The `eval()` function safely evaluates a string containing a Python literal.
        # It will turn the string "(3, 3, 1, 16)" into the tuple (3, 3, 1, 16).
        # It also handles the bias shapes like "-16", converting them to an integer.
        # We'll then convert the integer to a tuple.
        shape_val = eval(shape_str)
        
        if isinstance(shape_val, int):
            # Handle the bias case where shape is just "-16"
            # We want the shape to be a tuple, e.g., (16,)
            shape_tuple = (abs(shape_val),)
        else:
            # It's already a tuple, e.g., (3, 3, 1, 16)
            shape_tuple = shape_val
            
        param_shapes[varname] = shape_tuple

    model_args = []
    model_kwargs = dict()
    model_cls = cfg.model._target_.split(".")[-1]
    # if model_cls == "DWSModelForClassification":
    #     model_kwargs["weight_shapes"] = None
    #     model_kwargs["bias_shapes"] = None
    # elif model_cls in ("InvariantNFN", "StatNet"):
    #     data_sample = next(iter(train_loader))
    #     network_spec = network_spec_from_wsfeat(
    #         WeightSpaceFeatures(
    #             [wi.unsqueeze(1) for wi in data_sample.weights],
    #             [bi.unsqueeze(1) for bi in data_sample.biases],
    #         )
    #     )
    #     model_args.append(network_spec)
    model = hydra.utils.instantiate(cfg.model, *model_args, **model_kwargs).to(device)

    if rank == 0:
        logging.info(f"number of parameters: {count_parameters(model)}")

    if cfg.compile:
        model = torch.compile(model, **cfg.compile_kwargs)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = hydra.utils.instantiate(cfg.optim, params=parameters)
    if hasattr(cfg, "scheduler"):
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    else:
        scheduler = None

    criterion = hydra.utils.instantiate(cfg.loss)
    best_val_tau = -float("inf")
    best_val_loss = float("inf")
    best_val_l1_err = float("inf")
    best_test_results, best_val_results = None, None
    global_step, start_epoch = 0, 0 

    if cfg.load_ckpt:
        ckpt = torch.load(cfg.load_ckpt)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
        if "global_step" in ckpt:
            global_step = ckpt["global_step"]
        if rank == 0:
            logging.info(f"loaded checkpoint {cfg.load_ckpt}")
    if cfg.distributed:
        model = DDP(
            model, device_ids=cfg.distributed.device_ids, find_unused_parameters=False
        )
    model.train()

    if rank == 0:
        ckpt_dir = Path(hydra_cfg.runtime.output_dir) / wandb.run.path.split("/")[-1]
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler(**cfg.gradscaler)
    autocast_kwargs = dict(cfg.autocast)
    autocast_kwargs["dtype"] = getattr(torch, cfg.autocast.dtype, torch.float32)
    optimizer.zero_grad()
    epoch_iter = trange(start_epoch, cfg.n_epochs, disable=rank != 0)
    for epoch in epoch_iter:
        model.train()
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        for i, (graph_batch, original_params_batch) in enumerate(train_loader):
            graph_batch = graph_batch.to(device)
            original_params_batch = original_params_batch.to(device)
            gt_test_acc = graph_batch.y.to(device)
            optimizer.zero_grad()
            with torch.autocast(**autocast_kwargs):
                recon_params_flat = model(graph_batch)
                recon_params = unflatten_params_batch(recon_params_flat, original_set.layout, param_shapes)
                original_params = (original_params_batch.weights, original_params_batch.biases)
                batch_loss = torch.tensor(0.0, device=device)
                total_elements = 0 
                for j, (cifar_images, _) in enumerate(cifar10_test_loader):
                    cifar_images = cifar_images.to(device)
                    # Get logits for both batches of models using the current image batch
                    with torch.no_grad():
                        original_logits = get_logits_for_batch(original_params, cifar_images, cfg)
            
                    recon_logits = get_logits_for_batch(recon_params, cifar_images, cfg)
                    # Prepare logits for KLDivLoss
                    B_models, B_images, N_classes = recon_logits.shape
                    # The loss function expects input of shape (Number of Samples, Classes) 
                    # But we have (B_models, B_images, N_classes)
                    recon_logits_flat = recon_logits.view(-1, N_classes)
                    original_logits_flat = original_logits.view(-1, N_classes)
                    current_elements = B_models * B_images
                    # Calculate loss for this mini-batch of images and accumulate
                    batch_loss += criterion(recon_logits_flat, original_logits_flat) * current_elements
                    total_elements += current_elements

                avg_loss = batch_loss / total_elements
            
            scaler.scale(avg_loss).backward()
            if cfg.clip_grad:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, cfg.clip_grad_max_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if rank == 0:
                train_log = {
                    "train/loss": avg_loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                epoch_iter.set_postfix({
                    'train_loss': f'{avg_loss.item():.6f}'
                })
                if cfg.clip_grad and 'grad_norm' in locals():
                    train_log["grad_norm"] = grad_norm
                if scheduler is not None:
                    train_log["lr"] = scheduler.get_last_lr()[0]

                wandb.log(train_log)
            global_step += 1

        # --- Evaluation Step ---
        if (rank == 0):
            model.eval() # Set model to evaluation mode
            
            eval_args = {
                "net": model.module if cfg.distributed else model,
                "loader": val_loader,
                "criterion": criterion,
                "cifar10_test_loader": cifar10_test_loader,
                "device": device,
                "layout": original_set.layout,
                "param_shapes": param_shapes,
                "effective_conf": cfg,
            }
            val_results = evaluate(**eval_args)
            eval_args["loader"] = test_loader
            test_results = evaluate(**eval_args)

            val_tau = val_results["tau"]  # Update for progress bar
            val_l1_err = val_results["avg_err"]

            print(f"Epoch {epoch}, val L1 err: {val_l1_err:.9f}, val loss: {val_results['avg_loss']:.9f}, val Rsq: {val_results['rsq']:.9f}, val tau: {val_tau:.9f}")
            
            print(f"Epoch {epoch}, test L1 err: {test_results['avg_err']:.9f}, test loss: {test_results['avg_loss']:.9f}, test Rsq: {test_results['rsq']:.9f}, test tau: {test_results['tau']:.9f}")
            if cfg.get('save_checkpoints', True):
                if val_l1_err <= best_val_l1_err:
                    best_val_l1_err = val_l1_err
                    best_val_results = val_results
                    best_test_results = test_results
                    torch.save({
                            "model": (model.module if cfg.distributed else model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "cfg": cfg, "global_step": global_step,
                        }, ckpt_dir / "best_val.ckpt",
                    )
                if epoch == 0:
                    torch.save({
                            "model": (model.module if cfg.distributed else model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "cfg": cfg, "global_step": global_step,
                        }, ckpt_dir / "latest.ckpt",
                    )
                elif (cfg.eval_every % epoch == 0 or epoch == cfg.n_epochs - 1):
                    torch.save({
                            "model": (model.module if cfg.distributed else model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "cfg": cfg, "global_step": global_step,
                        }, ckpt_dir / "latest.ckpt",
                    )

            plt.clf()
            plt.scatter(val_results["actual"], val_results["pred"])
            plt.xlabel("Actual model accuracy")
            plt.ylabel("Predicted model accuracy")

            eval_log = {
                "val/loss": val_results["avg_loss"],
                "val/l1_err": val_results["avg_err"],
                "val/rsq": val_results["rsq"],
                "val/kendall_tau": val_results["tau"],
                "val/scatter": wandb.Image(plt),
                "test/loss": test_results["avg_loss"],
                "test/l1_err": test_results["avg_err"],
                "test/rsq": test_results["rsq"],
                "test/kendall_tau": test_results["tau"],
            }
            if best_val_results:
                eval_log["val/best_loss"] = best_val_results["avg_loss"]
                eval_log["val/best_l1_err"] = best_val_results["avg_err"]
                eval_log["val/best_tau"] = best_val_results["tau"]
            if best_test_results:
                eval_log["test/best_loss_at_val"] = best_test_results["avg_loss"]
                eval_log["test/best_l1_err_at_val"] = best_test_results["avg_err"]
                eval_log["test/best_tau_at_val"] = best_test_results["tau"]

            wandb.log(eval_log)
            model.train() # Set model back to training mode


def train_ddp(rank, cfg, hydra_cfg):
    ddp_setup(rank, cfg.distributed.world_size)
    cfg.distributed.rank = rank
    train(cfg, hydra_cfg)
    destroy_process_group()


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    if cfg.distributed:
        mp.spawn(
            train_ddp,
            args=(cfg, hydra_cfg),
            nprocs=cfg.distributed.world_size,
            join=True,
        )
    else:
        train(cfg, hydra_cfg)


if __name__ == "__main__":
    main()
