import torch
import torch_geometric
import yaml
import random
import torch.nn.functional as F
import torch.nn as nn
import os
import pathlib 
import torchvision
import torchvision.transforms as transforms
from src.data.cifar10_dataset import CNNBatch
from src.data import dataset
from tqdm import tqdm
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.scalegmn.autoencoder import get_autoencoder
from src.scalegmn.cnn import unflatten_params_batch, get_logits_for_batch
from src.utils.loss import select_criterion, KnowledgeDistillationLoss
from src.utils.optim import setup_optimization
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed, mask_input, mask_hidden, count_named_parameters
from src.scalegmn.utils import get_cifar10_test_loader, get_cifar10_train_loader
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import wandb
from scipy.optimize import linear_sum_assignment
from typing import List
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

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

def run_interpolation_for_pair(graph_batch, original_cnn_batch, net, args, eval_args, param_shapes, test_set_interpolation_layout):
    """
    Runs all specified interpolation experiments for a single pair of models.
    """
    pair_results = {}
    
    # Move data for this pair to the correct device
    device = eval_args["device"]
    graph_batch = graph_batch.to(device)
    original_cnn_batch = original_cnn_batch.to(device)
    
    print(f"--- Interpolating pair with accuracies: {graph_batch.y[0]:.4f} and {graph_batch.y[1]:.4f} ---")

    # --- ORIGINAL SPACE ---
    if 'original' in args.interpolation_methods:
        results_original = interpolate_and_evaluate_structured(
            cnn_batch_for_interp=original_cnn_batch, **eval_args
        )
        pair_results['Original Space'] = results_original

    # --- RECONSTRUCTED SPACE ---
    if 'reconstructed' in args.interpolation_methods:
        recon_flat_params_batch = net(graph_batch)
        recon_cnn_batch = CNNBatch(*unflatten_params_batch(recon_flat_params_batch, test_set_interpolation_layout, param_shapes), y=None)
        results_reconstructed = interpolate_and_evaluate_structured(
            cnn_batch_for_interp=recon_cnn_batch, **eval_args
        )
        pair_results['Reconstructed Space'] = results_reconstructed

    # --- LINEAR ASSIGNMENT SPACE ---
    if 'linear_assignment' in args.interpolation_methods:
        weights_A = [w[0] for w in original_cnn_batch.weights]
        biases_A = [b[0] for b in original_cnn_batch.biases]
        weights_B = [w[1] for w in original_cnn_batch.weights]
        biases_B = [b[1] for b in original_cnn_batch.biases]
        
        rebased_weights_B, rebased_biases_B = match_cnns_rebasin_einsum(weights_A, biases_A, weights_B, biases_B)
        
        aligned_weights = [torch.stack([wA, wB_rebased]) for wA, wB_rebased in zip(weights_A, rebased_weights_B)]
        aligned_biases = [torch.stack([bA, bB_rebased]) for bA, bB_rebased in zip(biases_A, rebased_biases_B)]
        aligned_cnn_batch = CNNBatch(weights=aligned_weights, biases=aligned_biases, y=None)
        
        results_aligned = interpolate_and_evaluate_structured(
            cnn_batch_for_interp=aligned_cnn_batch, **eval_args
        )
        pair_results['Linearly Aligned Space'] = results_aligned
        
    return pair_results

def apply_random_permutation(
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
    perm_dims: List[int],
    device: torch.device
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Applies a random permutation to the hidden layers of a CNN while preserving
    its function, to create a ground-truth test case for re-basing algorithms.

    Args:
        weights (List[torch.Tensor]): The original model weights.
        biases (List[torch.Tensor]): The original model biases.
        perm_dims (List[int]): A list of the dimensions of the permutable spaces.
                               For a model with dims [1, 16, 16, 16, 10], this
                               should be [16, 16, 16].
        device (torch.device): The device to create tensors on.

    Returns:
        A tuple of (permuted_weights, permuted_biases).
    """
    num_layers = len(weights)
    
    # Create random permutations for each hidden layer space.
    # The first (input) and last (output) spaces are not permuted.
    perms = [torch.arange(1, device=device)] # Input permutation is identity
    for dim in perm_dims:
        perms.append(torch.randperm(dim, device=device))
    perms.append(torch.arange(weights[-1].shape[0], device=device)) # Output permutation is identity
    
    print("\n--- Generating a randomly permuted, functionally identical model for sanity check ---")
    for i, p in enumerate(perms):
        if i > 0 and i < len(perms)-1:
            print(f"  - Random permutation for hidden space {i} (size {len(p)}) created.")

    permuted_weights = []
    permuted_biases = []

    for l in range(num_layers):
        P_in = perms[l]
        P_out = perms[l + 1]
        
        # Apply the compensatory permutations
        w_perm = torch.index_select(weights[l], 0, P_out)
        b_perm = torch.index_select(biases[l], 0, P_out)
        w_perm = torch.index_select(w_perm, 1, P_in)
        
        permuted_weights.append(w_perm)
        permuted_biases.append(b_perm)
            
    return permuted_weights, permuted_biases

# Helper to convert permutation indices to a matrix if needed, although direct indexing is better
def permutation_to_matrix(perm: np.ndarray) -> np.ndarray:
    mat = np.eye(len(perm), dtype=np.float64)
    return mat[perm, :]

def match_cnns_rebasin_einsum(
    weights_A: List[torch.Tensor],
    biases_A:  List[torch.Tensor],
    weights_B: List[torch.Tensor],
    biases_B:  List[torch.Tensor],
    max_iter: int = 100,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Finds optimal permutations for a simple sequential CNN using Einstein summation
    to avoid explicit flattening of kernels. This is functionally identical
    to the reshape-based approach.
    """
    # 1) SETUP: (Identical to before)
    assert len(weights_A) == len(weights_B) == len(biases_A) == len(biases_B)
    device, dtype = weights_A[0].device, weights_A[0].dtype

    W_A = [w.cpu().double().numpy() for w in weights_A]
    W_B = [w.cpu().double().numpy() for w in weights_B]
    b_A = [b.cpu().double().numpy() for b in biases_A]
    b_B = [b.cpu().double().numpy() for b in biases_B]

    num_layers = len(W_A)
    dims = [W_A[0].shape[1]] + [w.shape[0] for w in W_A]
    perms = [np.arange(d) for d in dims]

    print("Matching CNNs using the iterative Re-Basin algorithm (einsum version)...")

    # 2) ITERATIVE OPTIMIZATION
    for iteration in range(max_iter):
        changed = False
        
        for l in np.random.permutation(range(1, num_layers)):
            perm_dim = dims[l]
            C = np.zeros((perm_dim, perm_dim))

            # --- A) Cost from INCOMING weights (layer l-1) ---
            wA_in, bA_in = W_A[l-1], b_A[l-1]
            wB_in, bB_in = W_B[l-1], b_B[l-1]
            
            perm_in_indices = perms[l-1]
            if wB_in.ndim > 2: # Conv Layer
                wB_in_permuted = wB_in[:, perm_in_indices, :, :]
                C += np.einsum('oikl,jikl->oj', wA_in, wB_in_permuted, optimize=True)
            else: # Linear Layer (no flattening needed here either)
                wB_in_permuted = wB_in[:, perm_in_indices]
                C += wA_in @ wB_in_permuted.T # Standard matmul is fine for 2D

            C += np.outer(bA_in, bB_in)

            # --- B) Cost from OUTGOING weights (layer l) ---
            wA_out, wB_out = W_A[l], W_B[l]
            perm_out_indices = perms[l+1]
            if wB_out.ndim > 2: # Conv
                wB_out_permuted = wB_out[perm_out_indices, :, :, :]
                C += np.einsum('oikl,ojkl->ij', wA_out, wB_out_permuted, optimize=True)
            else: # Linear
                wB_out_permuted = wB_out[perm_out_indices, :]
                C += wA_out.T @ wB_out_permuted # Standard matmul is fine for 2D

            # Solve assignment and update permutations
            _, col_ind = linear_sum_assignment(C, maximize=True)
            
            if not np.array_equal(perms[l], col_ind):
                perms[l] = col_ind
                changed = True

        if not changed:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print(f"Warning: Max iterations ({max_iter}) reached without convergence.")

    # 3) FINAL APPLICATION OF PERMUTATIONS (This part is unchanged and correct)
    W_B_aligned = [torch.from_numpy(w).to(device).to(dtype) for w in W_B]
    b_B_aligned = [torch.from_numpy(b).to(device).to(dtype) for b in b_B]

    for l in range(num_layers):
        P_in_indices = torch.from_numpy(perms[l]).to(device)
        P_out_indices = torch.from_numpy(perms[l + 1]).to(device)
        
        W_B_aligned[l] = torch.index_select(W_B_aligned[l], 0, P_out_indices)
        b_B_aligned[l] = torch.index_select(b_B_aligned[l], 0, P_out_indices)
        W_B_aligned[l] = torch.index_select(W_B_aligned[l], 1, P_in_indices)
            
    return W_B_aligned, b_B_aligned

@torch.no_grad()
def evaluate_on_full_test_set(
    net,
    test_loader,
    cifar10_test_loader,
    device,
    layout,
    param_shapes,
    effective_conf,
):
    """
    Evaluates the autoencoder's accuracy prediction performance over the entire test set.

    This function exclusively calculates metrics related to how well the accuracy of
    reconstructed models matches the ground-truth accuracy of the original models.
    It does NOT calculate any reconstruction loss (e.g., KL divergence).

    Args:
        net (nn.Module): The trained autoencoder model.
        test_loader (DataLoader): DataLoader for the test set of model graphs.
        cifar10_test_loader (DataLoader): DataLoader for the CIFAR-10 test images.
        device (torch.device): The device to run evaluation on.
        layout (pd.DataFrame): DataFrame describing the CNN layer layout.
        param_shapes (dict): Dictionary of parameter shapes for unflattening.
        effective_conf (dict): The configuration dictionary.

    Returns:
        dict: A dictionary containing aggregated accuracy metrics: avg_err, rsq,
              tau, and the raw prediction/actual accuracy arrays.
    """
    net.eval()

    all_recon_accuracies = []
    all_teacher_accuracies = []

    # Loop over the full test set of models
    for graph_batch, original_params_batch in tqdm(test_loader, desc="Full Test Set Evaluation", leave=False):
        # Move data to the correct device
        graph_batch = graph_batch.to(device)
        original_params_batch = original_params_batch.to(device)
        
        # Store the ground-truth (teacher) accuracies for this batch
        all_teacher_accuracies.append(original_params_batch.y.cpu().numpy())

        # 1. Reconstruct model parameters from their graph representations
        recon_params_flat = net(graph_batch)
        recon_params = unflatten_params_batch(recon_params_flat, layout, param_shapes)
        
        # 2. Evaluate performance of the reconstructed models on the CIFAR-10 test set
        num_cifar_samples = 0
        B_models = recon_params[0][0].shape[0]
        recon_correct_preds = torch.zeros(B_models, device=device)

        for cifar_images, cifar_labels in cifar10_test_loader:
            cifar_images, cifar_labels = cifar_images.to(device), cifar_labels.to(device)
            num_cifar_samples += cifar_images.size(0)

            # Get raw logits ONLY for the reconstructed models
            recon_logits = get_logits_for_batch(recon_params, cifar_images, effective_conf)

            # Calculate accuracy of the reconstructed models
            _, predicted_class = torch.max(recon_logits, dim=2)  # Shape: (B_models, B_images)
            correct_mask = (predicted_class == cifar_labels.unsqueeze(0))
            recon_correct_preds += correct_mask.sum(dim=1)

        # --- Finalize accuracy for this batch of models ---
        batch_recon_accuracies = (recon_correct_preds / num_cifar_samples).cpu().numpy()
        all_recon_accuracies.append(batch_recon_accuracies)

    # --- Aggregate results from all batches ---
    pred = np.concatenate(all_recon_accuracies)
    actual = np.concatenate(all_teacher_accuracies)

    return {
        "avg_err": np.mean(np.abs(pred - actual)),  # L1 error
        "rsq": r2_score(actual, pred),
        "tau": kendalltau(actual, pred).correlation,
        "pred": pred,
        "actual": actual,
    }


@torch.no_grad()
def interpolate_and_evaluate_structured(
    cnn_batch_for_interp: CNNBatch,
    cifar10_test_loader,
    device,
    effective_conf,
    num_steps=21,
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
    accuracies, losses = [], []
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
# def plot_interpolation_results(results_original, results_reconstructed, metric='accuracies'):
#     # ... (code is correct from previous answers)
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(results_original['alphas'], results_original[metric], 'o-', label='Interpolation in Original Weight Space', color='tab:red')
#     ax.plot(results_reconstructed['alphas'], results_reconstructed[metric], 'x--', label='Interpolation in Reconstructed Weight Space', color='tab:green')
#     ax.scatter([0, 1], [results_original[metric][0], results_original[metric][-1]], s=120, marker='*', c='black', zorder=5, label='Original Model Performance')
#     ax.set_xlabel("Interpolation Coefficient (α)")
#     ax.set_ylabel(f"Model {metric.capitalize()} on CIFAR-10")
#     ax.set_title(f"Comparing Interpolation in Original vs. Reconstructed Weight Space")
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     plot_filename = f"interpolation_{metric}_comparison.png"
#     plt.savefig(plot_filename)
#     print(f"Saved interpolation plot to {plot_filename}")
#     plt.show()

def plot_interpolation_results(all_results, metric='accuracies'):
    """
    Plots interpolation results for multiple experiments on the same axes.

    Args:
        all_results (dict): A dictionary where keys are experiment names (e.g., 'Original')
                            and values are the results from `interpolate_and_evaluate_structured`.
        metric (str): The metric to plot ('accuracies' or 'losses').
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'Original Space': 'tab:red', 'Reconstructed Space': 'tab:green', 'Linearly Aligned Space': 'tab:blue'}
    
    # Separate markers and linestyles
    markers = {'Original Space': 'o', 'Reconstructed Space': 'x', 'Linearly Aligned Space': 's'}
    linestyles = {'Original Space': '-', 'Reconstructed Space': '--', 'Linearly Aligned Space': '-.'}

    for name, results in all_results.items():
        ax.plot(results['alphas'], results[metric], 
                linestyle=linestyles.get(name, '-'), 
                marker=markers.get(name, 'o'),
                color=colors.get(name, 'gray'),
                label=f'Interpolation in {name}')

    # Highlight the endpoints (original models), which should be common to all relevant curves
    if 'Original Space' in all_results:
        original_results = all_results['Original Space']
        ax.scatter([0, 1], [original_results[metric][0], original_results[metric][-1]], 
                   s=150, marker='*', c='black', zorder=5, label='Endpoint Model Performance')

    ax.set_xlabel("Interpolation Coefficient (α)")
    ax.set_ylabel(f"Model {metric.capitalize()} on CIFAR-10")
    ax.set_title(f"Comparison of Interpolation Paths ({metric.capitalize()})")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plot_filename = f"interpolation_comparison_{metric}.png"
    plt.savefig(plot_filename)
    print(f"Saved interpolation plot to {plot_filename}")
    plt.show()

def multiple_pairs():
    # --- 1. STANDARD SETUP (Boilerplate) ---
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))
    if conf.get("wandb", False):
        run = wandb.init(config=conf, **conf.get("wandb_args", {}))
        # Use wandb.config as the effective configuration
        effective_conf = wandb.config
    else:
        # If not using wandb, the original conf is the effective config
        effective_conf = conf
        run = None  # No wandb run object
    decoder_hidden_dim_list = [
        effective_conf["scalegmn_args"]["d_hid"] * elem
        for elem in effective_conf["decoder_args"]["d_hidden"]
    ]
    effective_conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list
    # Use 'effective_conf' consistently from here onwards
    torch.set_float32_matmul_precision("high")
    # print(yaml.dump(effective_conf, default_flow_style=False)) # Use effective_conf
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif (
        torch.cuda.is_available()
    ):  # Keep cuda check for cross-compatibility if needed elsewhere
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    # Wandb logging setup (already uses wandb.init implicitly if run exists)
    set_seed(effective_conf["train_args"]["seed"])  # Use effective_conf
    assert_symms(effective_conf)
    print(yaml.dump(effective_conf, default_flow_style=False))

    # =============================================================================================
    #   SETUP THE GRAPH DATASET FROM CNN AND DATALOADER
    # =============================================================================================
    equiv_on_hidden = mask_hidden(effective_conf)
    get_first_layer_mask = mask_input(effective_conf)

    # --- 2. DATASET SETUP ---
    # We are using a special dataset mode that you need to implement.
    # It should load the top 1500 models and sort them.
    test_set = dataset(
        effective_conf['data'],
        split='test',
        direction=effective_conf['scalegmn_args']['direction'],
        interpolation_many_pairs=True, 
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask
    )

    # Extract param_shapes once from the dataset's layout
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
        batch_size=effective_conf["batch_size"],
        num_workers=effective_conf["num_workers"],
    )

    # --- 3. LOAD AUTOENCODER MODEL ---
    effective_conf['scalegmn_args']["layer_layout"] = test_set.get_layer_layout()
    net = get_autoencoder(
        model_args=effective_conf,
        autoencoder_type=effective_conf["train_args"]["reconstruction_type"],
    )
    if effective_conf.get("checkpoint_path", None):
        checkpoint_path = pathlib.Path(effective_conf["checkpoint_path"])
        if checkpoint_path.exists():
            print(f"Loading model from {checkpoint_path}")
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(net)
    net.to(device).eval()

    # --- 4. NEW LOGIC: SAMPLE PAIRS & RUN EXPERIMENTS IN A LOOP ---
    print("\n" + "="*80)
    NUM_PAIRS = 20
    print(f"Sampling {NUM_PAIRS} random pairs from the top 1500 models...")
    sampled_pairs = sample_random_pairs(test_set, num_pairs=NUM_PAIRS, seed=effective_conf["train_args"]["seed"])
    
    # This dictionary will store lists of result curves for aggregation
    interpolation_methods_map = {
        'original': 'Original Space',
        'reconstructed': 'Reconstructed Space',
        'linear_assignment': 'Linearly Aligned Space'
    }
    aggregated_results = {exp_name: {'accuracies': [], 'losses': []} for exp_name in interpolation_methods_map.values()}

    eval_args = {
        "cifar10_test_loader": cifar10_subset_train_loader,
        "device": device,
        "effective_conf": effective_conf,
        "num_steps": 21
    }

    for i, (graph_batch, cnn_batch) in enumerate(sampled_pairs):
        print(f"\n===== Processing Pair {i+1}/{NUM_PAIRS} =====")
        
        # This function runs all interpolation methods for the given pair
        pair_results = run_interpolation_for_pair(graph_batch, cnn_batch, net, args, eval_args, param_shapes, test_set.layout)
        
        # Append the results of this pair to our aggregation dictionary
        for exp_name, results_dict in pair_results.items():
            if exp_name in aggregated_results:
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

    # --- 6. SAVE FINAL RESULTS ---
    if not final_averaged_results:
        print("No results were generated. Exiting.")
        return
        
    print("\nSaving final averaged results...")
    output_dir = pathlib.Path("analysis/resources/interpolation/cnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "interpolation_results.pt"  # The final, averaged file
    output_path = output_dir / filename  
    torch.save(final_averaged_results, output_path)
    print(f"All averaged interpolation results saved to {output_path}")

    # You can now run the plotting script on this final file.


def single_pair():
    # read config file
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))
    # Initialize W&B (if enabled)
    #    - W&B automatically merges sweep parameters with the provided 'config'.
    #    - 'wandb.config' will hold the final, merged configuration.
    if conf.get("wandb", False):
        run = wandb.init(config=conf, **conf.get("wandb_args", {}))
        # Use wandb.config as the effective configuration
        effective_conf = wandb.config
    else:
        # If not using wandb, the original conf is the effective config
        effective_conf = conf
        run = None  # No wandb run object
    decoder_hidden_dim_list = [
        effective_conf["scalegmn_args"]["d_hid"] * elem
        for elem in effective_conf["decoder_args"]["d_hidden"]
    ]
    effective_conf["decoder_args"]["d_hidden"] = decoder_hidden_dim_list
    # Use 'effective_conf' consistently from here onwards
    torch.set_float32_matmul_precision("high")
    # print(yaml.dump(effective_conf, default_flow_style=False)) # Use effective_conf
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif (
        torch.cuda.is_available()
    ):  # Keep cuda check for cross-compatibility if needed elsewhere
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    # Wandb logging setup (already uses wandb.init implicitly if run exists)
    set_seed(effective_conf["train_args"]["seed"])  # Use effective_conf
    assert_symms(effective_conf)
    print(yaml.dump(effective_conf, default_flow_style=False))

    # =============================================================================================
    #   SETUP THE GRAPH DATASET FROM CNN AND DATALOADER
    # =============================================================================================
    equiv_on_hidden = mask_hidden(effective_conf)
    get_first_layer_mask = mask_input(effective_conf)

    # test_set = dataset(
    #     effective_conf['data'],
    #     split='test',
    #     direction=effective_conf['scalegmn_args']['direction'],
    #     interpolation=False,  # This is for the original CNN batch
    #     equiv_on_hidden=equiv_on_hidden,
    #     get_first_layer_mask=get_first_layer_mask
    # )
    test_set_interpolation = dataset(effective_conf['data'],
                    split='test',
                    direction=effective_conf['scalegmn_args']['direction'],
                    interpolation=True,
                    equiv_on_hidden=equiv_on_hidden,
                    get_first_layer_mask=get_first_layer_mask)


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
    # Define the common generator for the 2 dataloaders (original CNN and graph)
    generator = torch.Generator().manual_seed(effective_conf["train_args"]["seed"])  # Use effective_conf
    # test_loader = torch_geometric.loader.DataLoader(
    #     dataset=test_set,
    #     batch_size=effective_conf["batch_size"],
    #     shuffle=False,
    #     num_workers=effective_conf["num_workers"],
    #     pin_memory=True,
    # )
    test_loader_interpolation = torch_geometric.loader.DataLoader(
        dataset=test_set_interpolation,
        batch_size=effective_conf["batch_size"],
        shuffle=False,
        num_workers=effective_conf["num_workers"],
        pin_memory=True,
    )
    cifar10_test_loader = get_cifar10_test_loader(
        batch_size=effective_conf["batch_size"],
        num_workers=effective_conf["num_workers"],
    )
    cifar10_subset_train_loader = get_cifar10_train_loader(
        batch_size=effective_conf["batch_size"],
        num_workers=effective_conf["num_workers"],
    )
    # =============================================================================================
    #   DEFINE  AND LOAD MODEL
    # =============================================================================================
    effective_conf['scalegmn_args']["layer_layout"] = test_set_interpolation.get_layer_layout()
    net = get_autoencoder(
        model_args=effective_conf,
        autoencoder_type=effective_conf["train_args"]["reconstruction_type"],
    )  # Use effective_conf
    
    #load net from path checkpoint if provided
    if effective_conf.get("checkpoint_path", None):
        checkpoint_path = pathlib.Path(effective_conf["checkpoint_path"])
        if checkpoint_path.exists():
            print(f"Loading model from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(state_dict)
        else:
            print(f"Checkpoint path {checkpoint_path} does not exist. Skipping loading.")
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
    #     effective_conf=effective_conf
    # )
    
    # print("\n--- Full Test Set Evaluation Results ---")
    # print(f"  L1 Error (Recon Acc vs. GT Acc): {test_results['avg_err']:.6f}")
    # print(f"  R-squared: {test_results['rsq']:.4f}")
    # print(f"  Kendall's Tau: {test_results['tau']:.4f}")
    # print("="*80 + "\n")
    print("\n" + "="*80)
    print("Starting interpolation analysis...")
    print(f"Running experiments: {args.interpolation_methods}")
    print("="*80 + "\n")
    all_results = {}
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
        "effective_conf": effective_conf,
        "num_steps": 21
    }
    # --- ORIGINAL SPACE ---
    if 'original' in args.interpolation_methods:
        print("\n--- Running: Interpolation in Original Space ---")
        results_original = interpolate_and_evaluate_structured(
            cnn_batch_for_interp=original_cnn_batch, **eval_args
        )
        all_results['Original Space'] = results_original
    # --- RECONSTRUCTED SPACE ---
    if 'reconstructed' in args.interpolation_methods:
        print("\n--- Running: Interpolation in Reconstructed Space (Autoencoder) ---")
        recon_flat_params_batch = net(graph_batch)
        recon_cnn_batch = CNNBatch(*unflatten_params_batch(recon_flat_params_batch, test_set_interpolation.layout, param_shapes), y=None)
        results_reconstructed = interpolate_and_evaluate_structured(
            cnn_batch_for_interp=recon_cnn_batch, **eval_args
        )
        all_results['Reconstructed Space'] = results_reconstructed

    # --- Experiment 3: Interpolation with Linear Assignment ---
    if 'linear_assignment' in args.interpolation_methods:
        print("\n--- Running: Interpolation in Linearly Aligned Space ---")
        # Extract individual models from the batch
        weights_A = [w[0] for w in original_cnn_batch.weights]
        biases_A = [b[0] for b in original_cnn_batch.biases]
        weights_B = [w[1] for w in original_cnn_batch.weights]
        biases_B = [b[1] for b in original_cnn_batch.biases]

        # Find the permutation for model B that aligns it with model A
        #rebased_weights_B, rebased_biases_B = match_cnns_rebasin_permutation_only(weights_A, biases_A, weights_B, biases_B)
        rebased_weights_B, rebased_biases_B = match_cnns_rebasin_einsum(weights_A, biases_A, weights_B, biases_B)

        # Create a new batch with model A and the rebased model B
        # Unsqueeze to add the batch dimension back
        aligned_weights = [torch.stack([wA, wB_rebased]) for wA, wB_rebased in zip(weights_A, rebased_weights_B)]
        aligned_biases = [torch.stack([bA, bB_rebased]) for bA, bB_rebased in zip(biases_A, rebased_biases_B)]
        aligned_cnn_batch = CNNBatch(weights=aligned_weights, biases=aligned_biases, y=None)
        
        results_aligned = interpolate_and_evaluate_structured(
            cnn_batch_for_interp=aligned_cnn_batch, **eval_args
        )
        all_results['Linearly Aligned Space'] = results_aligned
    
    # --- SANITY CHECK Experiment ---
    if 'sanity_check' in args.interpolation_methods:
        print("\n--- Running: Sanity Check (Aligning a model with its permuted self) ---")
        # 1. Get Model A's parameters
        weights_A = [w[0] for w in original_cnn_batch.weights]
        biases_A = [b[0] for b in original_cnn_batch.biases]

        # 2. Create Model B by randomly permuting Model A
        #    The permutable dimensions for your CNN are [16, 16, 16]
        perm_dims = [w.shape[0] for w in weights_A[0:-1]] # Gets the output dims of all but the last layer
        weights_B_permuted, biases_B_permuted = apply_random_permutation(
            weights_A, biases_A, perm_dims, device
        )
      
        # 3. Run the re-basing algorithm to try and align permuted B back to A
        print("\nAttempting to recover original alignment using re-basing algorithm...")
        rebased_weights_B, rebased_biases_B = match_cnns_rebasin_einsum(
            weights_A, biases_A, weights_B_permuted, biases_B_permuted
        )
        # 4. Check if the re-basing was successful
        # The re-based model B should now be almost identical to the original model A
        total_diff = 0
        for wA, wB_rebased in zip(weights_A, rebased_weights_B):
            total_diff += torch.sum(torch.abs(wA - wB_rebased))
        print(f"  - L1 difference between original A and re-based B: {total_diff.item():.6f}")
        if total_diff < 1e-4:
            print("  - SUCCESS: The re-basing algorithm successfully recovered the original alignment.")
        else:
            print("  - WARNING: The re-basing algorithm did NOT recover the original alignment. There may be a bug.")

        # 5. Create a new batch for interpolation
        aligned_weights = [torch.stack([wA, wB_rebased]) for wA, wB_rebased in zip(weights_A, rebased_weights_B)]
        aligned_biases = [torch.stack([bA, bB_rebased]) for bA, bB_rebased in zip(biases_A, rebased_biases_B)]
        aligned_cnn_batch = CNNBatch(weights=aligned_weights, biases=aligned_biases, y=None)
        
        results_sanity_check = interpolate_and_evaluate_structured(
            cnn_batch_for_interp=aligned_cnn_batch, **eval_args
        )
        # We'll rename this so it looks good on the plot
        all_results['Linearly Aligned Space'] = results_sanity_check





    # --- Plotting ---
    if not all_results:
        print("No interpolation methods were selected. Exiting.")
        return
    # plot_interpolation_results(results_original_space, results_reconstructed_space, metric='accuracies')
    # plot_interpolation_results(results_original_space, results_reconstructed_space, metric='losses')
    print("\n" + "="*80)
    # print("Generating plots...")
    # plot_interpolation_results(all_results, metric='accuracies')
    # plot_interpolation_results(all_results, metric='losses')
    print("Saving Results")
    output_dir = pathlib.Path("analysis/resources/interpolation/cnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "interpolation_results.pt"  
    output_path = output_dir / filename  
    torch.save(all_results, output_path)
    print(f"All interpolation results saved to {output_path}")


if __name__ == "__main__":
    arg_parser = setup_arg_parser()

    arg_parser.add_argument(
    '--interpolation_methods', 
    nargs='+', 
    default=['original', 'reconstructed', 'linear_assignment', 'sanity_check'],
    choices=['original', 'reconstructed', 'linear_assignment', 'sanity_check'],
    help='A list of interpolation methods to run and compare.'
    )
    arg_parser.add_argument(
    '--multiple_pairs',
    action='store_true',
    help='Whether to use multiple pairs of models for interpolation.'
    )
    # -----------------------------------------------------------
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    if args.multiple_pairs:
        multiple_pairs()
    else:
        single_pair()