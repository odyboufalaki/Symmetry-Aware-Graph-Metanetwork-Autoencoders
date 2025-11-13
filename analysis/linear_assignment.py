from typing import Callable
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def get_matching_function(name: str) -> Callable:
    matching_functions = {
        "PD": match_weights_biases_PD,
        "DP": match_weights_biases_DP,
        "P": match_weights_biases_P,
        "D": match_weights_biases_D
    }
    
    if name not in matching_functions:
        raise ValueError(f"Invalid matching function: {name}")
        
    return matching_functions[name]


def match_weights_biases_PD(
    weights_A: tuple[torch.Tensor, ...],
    weights_B: tuple[torch.Tensor, ...],
    biases_A: tuple[torch.Tensor, ...],
    biases_B: tuple[torch.Tensor, ...],
    max_iter: int = 1000,
):
    """
    Signed-permutation weight matching (Git Re-Basin with ± symmetry, PD form).

    Args:
        weights_A: tuple of weight matrices for models A
        weights_B: tuple of weight matrices for models B
        biases_A: tuple of biases for models A
        biases_B: tuple of biases for models B
        max_iter: maximum number of iterations

    Returns
        transformed_weights: tuple of transformed weight matrices for model B
        transformed_biases: tuple of transformed biases for model B
    """
    assert len(weights_A) == len(weights_B) == len(biases_A) == len(biases_B), \
        "All tuples must be of the same length"
    
    weights_A = [w.cpu().detach().numpy() for w in weights_A]
    weights_B = [w.cpu().detach().numpy() for w in weights_B]
    biases_A = [b.cpu().detach().numpy() for b in biases_A]
    biases_B = [b.cpu().detach().numpy() for b in biases_B]

    L = len(weights_A)
    layers_to_permute = list(range(1, L))
    dims = [weights_A[0].shape[1]] + [W.shape[0] for W in weights_A]
    perms = [np.arange(d) for d in dims]
    signs = [np.ones(d, dtype=int) for d in dims]

    def make_Q(l: int):
        P_matrix = np.eye(dims[l])[perms[l]]
        D_matrix = np.diag(signs[l])
        # return P_matrix
        return P_matrix @ D_matrix
        
    Q = [np.eye(dims[0])] + [make_Q(l) for l in range(1, L)] + [np.eye(dims[L])]

    for _ in range(max_iter):
        changed = False
        for l in np.random.permutation(layers_to_permute):
            C = (
                weights_A[l-1] @ Q[l-1] @ weights_B[l-1].T
                + weights_A[l].T @ Q[l+1] @ weights_B[l]
                + biases_A[l-1][:, np.newaxis] @ biases_B[l-1][np.newaxis, :]
            )
            cost_abs = np.abs(C)
            row_indices, col_indices = linear_sum_assignment(cost_abs, maximize=True)
            new_perm_for_layer = col_indices
            signs_from_C_elements = np.sign(C[row_indices, col_indices])
            new_signs_for_D_cols = np.ones_like(signs[l])
            new_signs_for_D_cols[new_perm_for_layer] = signs_from_C_elements

            if (new_perm_for_layer != perms[l]).any() or \
               (new_signs_for_D_cols != signs[l]).any():
                perms[l] = new_perm_for_layer
                signs[l] = new_signs_for_D_cols
                Q[l] = make_Q(l)
                changed = True
                
        if not changed:
            break

    transformed_weights = [None] * L
    transformed_biases = [None] * L

    for l_idx in range(L):
        P_l_idx_T = (np.eye(dims[l_idx])[perms[l_idx]]).T
        D_l_idx = np.diag(signs[l_idx])
        Q_l_idx_T = D_l_idx @ P_l_idx_T
        transformed_weights[l_idx] = Q[l_idx+1] @ weights_B[l_idx] @ Q_l_idx_T
        transformed_biases[l_idx] = Q[l_idx+1] @ biases_B[l_idx]

    return transformed_weights, transformed_biases


def match_weights_biases_DP(
    weights_A: tuple[torch.Tensor, ...],
    weights_B: tuple[torch.Tensor, ...],
    biases_A: tuple[torch.Tensor, ...],
    biases_B: tuple[torch.Tensor, ...],
    max_iter: int = 1000,
):
    """
    Signed-permutation weight matching (Git Re-Basin with ± symmetry).

    Args:
        weights_A: tuple of weight matrices for models A
        weights_B: tuple of weight matrices for models B
        biases_A: tuple of biases for models A
        biases_B: tuple of biases for models B
        max_iter: maximum number of iterations

    Returns
        Q : list of signed-permutation matrices that permute/flip WB onto WA
    """
    assert len(weights_A) == len(weights_B) == len(biases_A) == len(biases_B), "All tuples must be of the same length"
    
    # Convert to numpy arrays
    weights_A = [w.cpu().numpy() for w in weights_A]
    weights_B = [w.cpu().numpy() for w in weights_B]
    biases_A = [b.cpu().numpy() for b in biases_A]
    biases_B = [b.cpu().numpy() for b in biases_B]

    L = len(weights_A)
    input_layer, output_layer = 0, L
    layers = list(range(input_layer, output_layer))[1:]  # [1, 2]
    dims = [weights_A[0].shape[1]] + [W.shape[0] for W in weights_A]  # [2, 32, 32, 1]
    perms  = [np.arange(d) for d in dims]
    signs  = [np.ones(d, dtype=int) for d in dims]

    def make_Q(l):
        P = np.eye(dims[l])[perms[l]]
        D = np.diag(signs[l])
        return D @ P

    Q = [np.eye(dims[0])] + [make_Q(l) for l in layers] + [np.eye(dims[L])]  # Q[0] and Q[L] are identity

    for _ in range(max_iter):
        changed = False
        for l in np.random.permutation(layers):  # exclude endpoints
            C = (
                weights_A[l-1] @ Q[l-1] @ weights_B[l-1].T
                + weights_A[l].T @ Q[l+1] @ weights_B[l]
                + np.outer(biases_A[l-1], biases_B[l-1])
            )
            cost = np.abs(C)
            row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
            new_perm = col_ind
            new_signs = np.sign(C[row_ind, col_ind])
            if (new_perm != perms[l]).any() or (new_signs != signs[l]).any():
                perms[l] = new_perm
                signs[l] = new_signs
                Q[l] = make_Q(l)
                changed  = True
        if not changed:
            break

    transformed_weights = [None] * L
    transformed_biases = [None] * L

    # Apply the transformation to INR B
    for l in range(3):  # [0, 1, 2]
        # Current layer
        transformed_weights[l] = Q[l+1] @ weights_B[l] @ Q[l].T
        transformed_biases[l] = Q[l+1] @ biases_B[l]

    return transformed_weights, transformed_biases


def match_weights_biases_P(
    weights_A: tuple[torch.Tensor, ...],
    weights_B: tuple[torch.Tensor, ...],
    biases_A: tuple[torch.Tensor, ...],
    biases_B: tuple[torch.Tensor, ...],
    max_iter: int = 1000,
):
    """
    Permutation only weight matching (Git Re-Basin with ± symmetry).

    Args:
        weights_A: tuple of weight matrices for models A
        weights_B: tuple of weight matrices for models B
        biases_A: tuple of biases for models A
        biases_B: tuple of biases for models B
        max_iter: maximum number of iterations

    Returns
        P : list of permutation matrices that permute WB onto WA
    """
    assert len(weights_A) == len(weights_B) == len(biases_A) == len(biases_B), "All tuples must be of the same length"
    
    # Convert to numpy arrays
    weights_A = [w.cpu().numpy() for w in weights_A]
    weights_B = [w.cpu().numpy() for w in weights_B]
    biases_A = [b.cpu().numpy() for b in biases_A]
    biases_B = [b.cpu().numpy() for b in biases_B]

    L = len(weights_A)
    input_layer, output_layer = 0, L
    layers = list(range(input_layer, output_layer))[1:]  # [1, 2]
    dims = [weights_A[0].shape[1]] + [W.shape[0] for W in weights_A]  # [2, 32, 32, 1]
    perms  = [np.arange(d) for d in dims]

    def make_P(l):
        P = np.eye(dims[l])[perms[l]]
        return P

    P = [np.eye(dims[0])] + [make_P(l) for l in layers] + [np.eye(dims[L])]  # P[0] and P[L] are identity

    for _ in range(max_iter):
        changed = False
        for l in np.random.permutation(layers):  # exclude endpoints
            C = (
                weights_A[l-1] @ P[l-1] @ weights_B[l-1].T
                + weights_A[l].T @ P[l+1] @ weights_B[l]
                + np.outer(biases_A[l-1], biases_B[l-1])
            )
            _, col_ind = linear_sum_assignment(C, maximize=True)
            new_perm = col_ind
            if (new_perm != perms[l]).any():
                perms[l] = new_perm
                P[l] = make_P(l)
                changed  = True
        if not changed:
            break

    transformed_weights = [None] * L
    transformed_biases = [None] * L

    # Apply the transformation to INR B
    for l in range(3):  # [0, 1, 2]
        transformed_weights[l] = P[l+1] @ weights_B[l] @ P[l].T
        transformed_biases[l] = P[l+1] @ biases_B[l]

    return transformed_weights, transformed_biases


def match_weights_biases_D(
    weights_A: tuple[torch.Tensor, ...],
    weights_B: tuple[torch.Tensor, ...],
    biases_A: tuple[torch.Tensor, ...],
    biases_B: tuple[torch.Tensor, ...],
    max_iter: int = 1000,
):
    """
    Signed only weight matching (Git Re-Basin with ± symmetry).

    Args:
        weights_A: tuple of weight matrices for models A
        weights_B: tuple of weight matrices for models B
        biases_A: tuple of biases for models A
        biases_B: tuple of biases for models B
        max_iter: maximum number of iterations

    Returns
        D : list of signed-permutation matrices that permute/flip WB onto WA
    """
    assert len(weights_A) == len(weights_B) == len(biases_A) == len(biases_B), "All tuples must be of the same length"
    
    # Convert to numpy arrays
    weights_A = [w.cpu().numpy() for w in weights_A]
    weights_B = [w.cpu().numpy() for w in weights_B]
    biases_A = [b.cpu().numpy() for b in biases_A]
    biases_B = [b.cpu().numpy() for b in biases_B]

    L = len(weights_A)
    input_layer, output_layer = 0, L
    layers = list(range(input_layer, output_layer))[1:]  # [1, 2]
    dims = [weights_A[0].shape[1]] + [W.shape[0] for W in weights_A]  # [2, 32, 32, 1]
    perms  = [np.arange(d) for d in dims]
    signs  = [np.ones(d, dtype=int) for d in dims]

    def make_D(l):
        D = np.diag(signs[l])
        return D

    D = [np.eye(dims[0])] + [make_D(l) for l in layers] + [np.eye(dims[L])]  # D[0] and D[L] are identity

    for _ in range(max_iter):
        changed = False
        for l in np.random.permutation(layers):  # exclude endpoints
            C = (
                weights_A[l-1] @ D[l-1] @ weights_B[l-1].T
                + weights_A[l].T @ D[l+1] @ weights_B[l]
                + np.outer(biases_A[l-1], biases_B[l-1])
            )
            new_signs = np.sign(np.diag(C))
            if (new_signs != signs[l]).any():
                signs[l] = new_signs
                D[l] = make_D(l)
                changed  = True
        if not changed:
            break

    transformed_weights = [None] * L
    transformed_biases = [None] * L

    # Apply the transformation to INR B
    for l in range(3):  # [0, 1, 2]
        # Current layer
        transformed_weights[l] = D[l+1] @ weights_B[l] @ D[l].T
        transformed_biases[l] = D[l+1] @ biases_B[l]

    return transformed_weights, transformed_biases


def match_weights_biases_batch(
    weights_A_batch: tuple[torch.Tensor, ...],
    weights_B_batch: tuple[torch.Tensor, ...],
    biases_A_batch: tuple[torch.Tensor, ...],
    biases_B_batch: tuple[torch.Tensor, ...],
    max_iter: int = 1000,
    matching_type: str = "DP",
):
    """
    Match biases between two batches of models.
    
    Args:
        weights_A_batch: Tuple of weight tensors with shape (batch_size, dim)
        weights_B_batch: Tuple of weight tensors with shape (batch_size, dim) 
        biases_A_batch: Tuple of bias tensors with shape (batch_size, dim)
        biases_B_batch: Tuple of bias tensors with shape (batch_size, dim)
        max_iter: Maximum number of iterations for matching
        
    Returns:
        List of matched permutation matrices for each batch element
    """
    assert len(biases_A_batch) == len(biases_B_batch) == len(weights_A_batch) == len(weights_B_batch), "All tuples must be of the same length"
    
    batch_size = biases_A_batch[0].shape[0]
    all_transformed_weights = []
    all_transformed_biases = []
    
    for i in range(batch_size):
        # Extract single batch element
        biases_A_i = tuple(b[i].squeeze(-1) for b in biases_A_batch) 
        biases_B_i = tuple(b[i].squeeze(-1) for b in biases_B_batch)
        weights_A_i = tuple(w[i].squeeze(-1).T for w in weights_A_batch)
        weights_B_i = tuple(w[i].squeeze(-1).T for w in weights_B_batch)
        
        # Match single element
        matching_function = get_matching_function(matching_type)
        transformed_weights_i, transformed_biases_i = matching_function(
            weights_A=weights_A_i,
            weights_B=weights_B_i,
            biases_A=biases_A_i,
            biases_B=biases_B_i,
            max_iter=max_iter
        )
        all_transformed_weights.append(transformed_weights_i)
        all_transformed_biases.append(transformed_biases_i)
        
    return all_transformed_weights, all_transformed_biases