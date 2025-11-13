from src.data.cifar10_dataset import CNNBatch
from src.utils.loss import KnowledgeDistillationLoss
import logging
import random
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
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

from src.neural_graphs.experiments.utils import (
    count_parameters,
    set_logger,
    set_seed,
)
from src.neural_graphs.experiments.inr_classification.main import ddp_setup
from src.neural_graphs.nn.nfn.common.data import WeightSpaceFeatures, network_spec_from_wsfeat
from src.scalegmn.cnn import unflatten_params_batch, get_logits_for_batch
from src.scalegmn.utils import get_cifar10_test_loader, get_cifar10_train_loader

#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
OmegaConf.register_new_resolver("prod", lambda x, y: x * y)


def sample_random_pairs(dataset, num_pairs=10, seed=0):
    """
    Samples N random, non-overlapping pairs of models from a pre-sorted dataset.
    Since the dataset is sorted by accuracy, for any pair of indices (i, j)
    where i < j, it's guaranteed that acc(i) <= acc(j).

    Args:
        dataset (Dataset): A dataset where each item is a tuple (graph_data, cnn_data),
                           PRE-SORTED by accuracy in ascending order.
        num_pairs (int): The number of pairs to sample.
        seed (int): A random seed for reproducibility.

    Returns:
        List[Tuple]: A list containing N tuples, where each tuple is a valid pair
                     of (graph_batch, original_cnn_batch).
    """
    rng = random.Random(seed)
    num_models = len(dataset)
    
    if num_models < 2:
        raise ValueError(f"Dataset has {num_models} models, but at least 2 are required to form a pair.")

    sampled_pairs = []
    used_indices = set()

    attempts = 0
    max_attempts = num_pairs * 5 # Failsafe to prevent infinite loops

    while len(sampled_pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        
        # 1. Sample two distinct indices from the range [0, num_models - 1]
        idx1, idx2 = rng.sample(range(num_models), 2)

        # 2. Ensure we haven't already used these models in another pair
        if idx1 in used_indices or idx2 in used_indices:
            continue

        # 3. Enforce the order idx1 < idx2 to ensure acc1 < acc2
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1 # Swap them

        # 4. Get the models from the dataset
        graph1, cnn1 = dataset[idx1]
        graph2, cnn2 = dataset[idx2]

        # 6. Manually collate the two models into a batch of size 2
        graph_batch = torch_geometric.data.Batch.from_data_list([graph1, graph2])
        
        weights = [torch.stack([w1, w2]) for w1, w2 in zip(cnn1.weights, cnn2.weights)]
        biases = [torch.stack([b1, b2]) for b1, b2 in zip(cnn1.biases, cnn2.biases)]
        y = torch.tensor([graph1.y, graph2.y])
        cnn_batch = CNNBatch(weights=tuple(weights), biases=tuple(biases), y=y)

        sampled_pairs.append((graph_batch, cnn_batch))
        used_indices.add(idx1)
        used_indices.add(idx2)

    if len(sampled_pairs) < num_pairs:
        print(f"Warning: Only able to sample {len(sampled_pairs)} valid pairs after {max_attempts} attempts. Requested {num_pairs}.")
        
    return sampled_pairs


def run_interpolation_for_pair_ng(graph_batch, original_cnn_batch, net, eval_args, param_shapes, dataset_layout):
    """
    Runs interpolation for Original and Reconstructed spaces for a single pair of models.
    """
    pair_results = {}
    device = eval_args["device"]
    graph_batch = graph_batch.to(device)
    original_cnn_batch = original_cnn_batch.to(device)
    
    print(f"--- Interpolating pair with accuracies: {graph_batch.y[0]:.4f} and {graph_batch.y[1]:.4f} ---")

    # --- ORIGINAL SPACE ---
    results_original = interpolate_and_evaluate_structured(
        cnn_batch_for_interp=original_cnn_batch, **eval_args
    )
    pair_results['Original Space'] = results_original

    # --- RECONSTRUCTED SPACE ---
    recon_flat_params_batch = net(graph_batch)
    recon_cnn_batch = CNNBatch(*unflatten_params_batch(recon_flat_params_batch, dataset_layout, param_shapes), y=None)
    results_reconstructed = interpolate_and_evaluate_structured(
        cnn_batch_for_interp=recon_cnn_batch, **eval_args
    )
    pair_results['Reconstructed Space'] = results_reconstructed
        
    return pair_results


def multiple_pairs_main(cfg):
    """
    Main workflow for sampling multiple pairs, running interpolation, and averaging results.
    """
    # --- 1. BOILERPLATE SETUP ---
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. DATASET SETUP ---
    # This assumes your hydra config for the dataset now includes `interpolation_many_pairs=True`
    test_set = hydra.utils.instantiate(cfg.data.test)
    
    param_shapes = {}
    for _, row in test_set.layout.iterrows():
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

    cifar10_subset_train_loader = get_cifar10_train_loader(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    # --- 3. MODEL LOADING ---
    net = hydra.utils.instantiate(cfg.model).to(device)
    if cfg.load_ckpt:
        ckpt = torch.load(cfg.load_ckpt, map_location=device)
        net.load_state_dict(ckpt["model"])
        logging.info(f"Loaded checkpoint {cfg.load_ckpt}")
    net.eval()

    # --- 4. SAMPLING, LOOPING, AND AGGREGATING ---
    NUM_PAIRS = 20
    print(f"\nSampling {NUM_PAIRS} random pairs from the dataset...")
    sampled_pairs = sample_random_pairs(test_set, num_pairs=NUM_PAIRS, seed=cfg.seed)

    aggregated_results = {
        'Original Space': {'accuracies': [], 'losses': []},
        'Reconstructed Space': {'accuracies': [], 'losses': []}
    }

    eval_args = {
        "cifar10_test_loader": cifar10_subset_train_loader,
        "device": device,
        "effective_conf": cfg,
        "num_steps": 21
    }

    for i, (graph_batch, cnn_batch) in enumerate(sampled_pairs):
        print(f"\n===== Processing Pair {i+1}/{NUM_PAIRS} =====")
        pair_results = run_interpolation_for_pair_ng(
            graph_batch, cnn_batch, net, eval_args, param_shapes, test_set.layout
        )
        for exp_name, results_dict in pair_results.items():
            aggregated_results[exp_name]['accuracies'].append(np.array(results_dict['accuracies']))
            aggregated_results[exp_name]['losses'].append(np.array(results_dict['losses']))

    # --- 5. AVERAGE THE RESULTS ---
    print("\n" + "="*80)
    print("Averaging results across all pairs...")
    final_averaged_results = {}

    for exp_name, metrics_dict in aggregated_results.items():
        if not metrics_dict['accuracies']: continue

        # Create the raw [N, T] matrices
        raw_accuracies_matrix = np.stack(metrics_dict['accuracies'])
        raw_losses_matrix = np.stack(metrics_dict['losses'])

        # Calculate the mean curves
        avg_accuracies = np.mean(raw_accuracies_matrix, axis=0)
        avg_losses = np.mean(raw_losses_matrix, axis=0)
        
        # Store both the mean and the raw data
        final_averaged_results[exp_name] = {
            'alphas': np.linspace(0, 1, len(avg_accuracies)).tolist(),
            'accuracies': avg_accuracies.tolist(),
            'losses': avg_losses.tolist(),
            'raw_accuracies': raw_accuracies_matrix.tolist(), # Add raw matrix
            'raw_losses': raw_losses_matrix.tolist()         # Add raw matrix
        }
        print(f"  - Processed '{exp_name}' curves with shape {raw_accuracies_matrix.shape}")

    # --- 6. SAVING FINAL RESULTS ---
    if 'Reconstructed Space' not in final_averaged_results:
        print("No reconstructed space results were generated. Exiting.")
        return
        
    print("\nSaving final averaged results for Reconstructed Space...")
    output_dir = Path("analysis/resources/interpolation/cnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "interpolation_results_ng.pt"
    output_path = output_dir / filename
    # Save only the reconstructed curve, as expected by the plotting script
    torch.save(final_averaged_results['Reconstructed Space'], output_path)
    print(f"Averaged interpolation results saved to {output_path}")


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


@torch.no_grad()
def interpolate_and_evaluate_structured(
    cnn_batch_for_interp: CNNBatch,
    cifar10_test_loader,
    device,
    effective_conf,
    num_steps=11,
):
    """
    Performs direct, structured interpolation between two models within a CNNBatch.
    
    Args:
        cnn_batch_for_interp (CNNBatch): A batch object containing exactly two models.
        ... (other args for evaluation)
    """
    # The __len__ of CNNBatch correctly returns the batch size.
    batch_size = len(cnn_batch_for_interp)
    

    alphas = np.linspace(0, 1, num_steps)
    accuracies = []
    losses = []
    
    ce_criterion = nn.CrossEntropyLoss()

    # --- CORRECTLY EXTRACTING THE TWO MODELS ---
    # cnn_batch_for_interp.weights is a TUPLE of LAYER tensors.
    # Each tensor has shape (B, ...), where B is the batch size (2).
    
    # To get the parameters for the FIRST model (index 0 in the batch):
    # We iterate through each layer's batched tensor and take the 0-th slice.
    model1_weights = [layer_tensor[0] for layer_tensor in cnn_batch_for_interp.weights]
    model1_biases  = [layer_tensor[0] for layer_tensor in cnn_batch_for_interp.biases]
    
    # To get the parameters for the SECOND model (index 1 in the batch):
    # We iterate through each layer's batched tensor and take the 1st slice.
    model2_weights = [layer_tensor[1] for layer_tensor in cnn_batch_for_interp.weights]
    model2_biases  = [layer_tensor[1] for layer_tensor in cnn_batch_for_interp.biases]

    for alpha in tqdm(alphas, desc="Interpolating models"):
        # 1. Create a NEW, SINGLE interpolated model for this alpha value
        interp_weights = [(1 - alpha) * w1 + alpha * w2 for w1, w2 in zip(model1_weights, model2_weights)]
        interp_biases  = [(1 - alpha) * b1 + alpha * b2 for b1, b2 in zip(model1_biases, model2_biases)]

        # get_logits_for_batch expects a batch. We create a "batch of 1"
        # for our single interpolated model by unsqueezing the batch dimension.
        interp_model_params = (
            [w.unsqueeze(0) for w in interp_weights],
            [b.unsqueeze(0) for b in interp_biases]
        )
        
        # 2. Evaluate the performance of this single interpolated model
        total_correct, total_loss, num_samples = 0, 0, 0
        for images, labels in cifar10_test_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits_batch = get_logits_for_batch(interp_model_params, images, effective_conf)
            logits = logits_batch.squeeze(0)
            
            loss = ce_criterion(logits, labels)
            _, predicted = torch.max(logits, 1)
            
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
        accuracies.append(total_correct / num_samples)
        losses.append(total_loss / num_samples)
        
    return {"alphas": alphas, "accuracies": accuracies, "losses": losses}

# The plotting function is unchanged and correct.
def plot_interpolation_results(results_original, results_reconstructed, metric='accuracies'):
    # ... (code is correct from previous answers)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_original['alphas'], results_original[metric], 'o-', label='Interpolation in Original Weight Space', color='tab:red')
    ax.plot(results_reconstructed['alphas'], results_reconstructed[metric], 'x--', label='Interpolation in Reconstructed Weight Space', color='tab:green')
    ax.scatter([0, 1], [results_original[metric][0], results_original[metric][-1]], s=120, marker='*', c='black', zorder=5, label='Original Model Performance')
    ax.set_xlabel("Interpolation Coefficient (α)")
    ax.set_ylabel(f"Model {metric.capitalize()} on CIFAR-10")
    ax.set_title(f"Comparing Interpolation in Original vs. Reconstructed Weight Space")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plot_filename = f"interpolation_{metric}_comparison_ng.png"
    plt.savefig(plot_filename)
    print(f"Saved interpolation plot to {plot_filename}")
    plt.show()


def single_pair_main(cfg):

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        set_seed(cfg.seed)
    
    rank = OmegaConf.select(cfg, "distributed.rank", default=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_set_interpolation = hydra.utils.instantiate(cfg.data.test)


    param_shapes = {}
    for _, row in test_set_interpolation.layout.iterrows():
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

    
    print(f'Len Interpolation set: {len(test_set_interpolation)}')
    
    test_loader_interpolation = torch_geometric.loader.DataLoader(
            dataset=test_set_interpolation,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    cifar10_test_loader = get_cifar10_test_loader(
        batch_size=cfg.batch_size,
        num_workers= cfg.num_workers,
    )

    cifar10_subset_train_loader = get_cifar10_train_loader(
        batch_size=cfg.batch_size,
        num_workers = cfg.num_workers,
    )
 
    model_args = []
    model_kwargs = dict()
    model_cls = cfg.model._target_.split(".")[-1]

    net = hydra.utils.instantiate(cfg.model, *model_args, **model_kwargs).to(device)
    
    #load net from path checkpoint if provided
    if cfg.load_ckpt:
        ckpt = torch.load(cfg.load_ckpt)
        net.load_state_dict(ckpt["model"])
    if rank == 0:
        logging.info(f"loaded checkpoint {cfg.load_ckpt}")
    print(net)
    
    net.to(device).eval()

    # =============================================================================================
    #   RUN FULL TEST SET EVALUATION (ACCURACY-ONLY)
    # =============================================================================================
    # print("\n" + "="*80)
    # print("Running accuracy-focused evaluation on the full test set...")
    
    # # The new function no longer requires a criterion/loss function
    # test_results = evaluate_on_full_test_set(
    #     net=net,
    #     test_loader=test_loader,
    #     cifar10_test_loader=cifar10_test_loader,
    #     device=device,
    #     layout=test_set.layout,
    #     param_shapes=param_shapes,
    #     effective_conf=cfg
    # )
    
    # print("\n--- Full Test Set Evaluation Results ---")
    # print(f"  L1 Error (Recon Acc vs. GT Acc): {test_results['avg_err']:.6f}")
    # print(f"  R-squared: {test_results['rsq']:.4f}")
    # print(f"  Kendall's Tau: {test_results['tau']:.4f}")
    # print("="*80 + "\n")
    
    if cfg.compile:
        net = torch.compile(net, **cfg.compile_kwargs)

    criterion = hydra.utils.instantiate(cfg.loss)

    print("Proceeding with interpolation analysis...")
    # Get a single batch of two models from the test loader
    try:
        graph_batch, original_cnn_batch = next(iter(test_loader_interpolation))
    except StopIteration:
        print("ERROR: Test loader is empty. Cannot perform interpolation.")
        return

    graph_batch = graph_batch.to(device)
    original_cnn_batch = original_cnn_batch.to(device)

    print(f"Interpolating between Model 1 (Acc: {graph_batch.y[0]:.4f}) and Model 2 (Acc: {graph_batch.y[1]:.4f})")
    
    eval_args = {
        "cifar10_test_loader": cifar10_subset_train_loader,
        "device": device,
        "effective_conf": cfg,
        "num_steps": 21
    }

    # --- ORIGINAL SPACE ---
    results_original_space = interpolate_and_evaluate_structured(
        cnn_batch_for_interp=original_cnn_batch, **eval_args
    )

    # --- RECONSTRUCTED SPACE ---
    recon_flat_params_batch = net(graph_batch)
    recon_cnn_batch = CNNBatch(*unflatten_params_batch(recon_flat_params_batch, test_set_interpolation.layout, param_shapes), y=None)
    
    results_reconstructed_space = interpolate_and_evaluate_structured(
        cnn_batch_for_interp=recon_cnn_batch, **eval_args
    )
    
    # --- Plotting ---
    #plot_interpolation_results(results_original_space, results_reconstructed_space, metric='accuracies')
    #plot_interpolation_results(results_original_space, results_reconstructed_space, metric='losses')

    print("\n" + "="*80)
    print("Saving Reconstructed Space results...")
    output_dir = Path("analysis/resources/interpolation/cnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "interpolation_results_ng.pt"  
    output_path = output_dir / filename  
    torch.save(results_reconstructed_space, output_path)
    print(f"Reconstructed space interpolation results saved to {output_path}")
    print("="*80)


@hydra.main(config_path="../src/neural_graphs/experiments/cnn_generalization/configs", config_name="base", version_base=None)
def main(cfg):
    """
    Main entry point that dispatches to the correct workflow based on config.
    """
    if cfg.get("interpolation_many_pairs", False):
        print("--- Running in MULTI-PAIR averaging mode ---")
        multiple_pairs_main(cfg)
    else:
        print("--- Running in SINGLE-PAIR mode ---")
        single_pair_main(cfg)


if __name__ == "__main__":
    main()