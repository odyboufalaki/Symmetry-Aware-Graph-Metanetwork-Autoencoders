from functools import partial
import json
import torch
import argparse
import os
import matplotlib.pyplot as plt
import torch_geometric
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Union, NamedTuple
import yaml
from src.utils.helpers import overwrite_conf, set_seed, assert_symms, mask_hidden, mask_input
from src.scalegmn.autoencoder import get_autoencoder
from src.data.cifar10_dataset import NFNZooDataset, CNNBatch
from src.data import dataset
from src.scalegmn.utils import get_cifar10_test_loader
from src.scalegmn.cnn import unflatten_params_batch, cnn_functional_forward, get_logits_for_batch
from tqdm import tqdm
import shutil

# DONT USE THIS FILE

# Variables
Tensor = torch.Tensor
StateDict = Dict[str, Tensor]
Transform = Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
CNN_LAYER_ARCHITECTURE = [1, 16, 16, 16, 10]
CONV_NUMBER_LAYERS = len(CNN_LAYER_ARCHITECTURE) - 2
LINEAR_NUMBER_LAYERS = 1

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_size",
        type=int,
        default=4,
        help="Number of augmented CNNs to generate",
    )
    p.add_argument(
        "--conf",
        type=str,
        default="./configs/cifar10_rec/scalegmn_relu.yml",
        help="Path to the configuration file",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with a smaller dataset",
    )
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument(
         "--transform_type",
         type=str,
         default="PD",
         choices=["PD", "P", "D"],
         help="Type of transformation to apply to the weights and biases",
     )
    p.add_argument("--seed", type=int, default=100)
    return p.parse_args()


# ------------------------------
# Group transformation functions
# def get_transform_function(name: str) -> Callable:
#     transform_functions = {
#         "PD": partial(transform_weights_biases, flip_signs=True, permute=True),
#         "P": partial(transform_weights_biases, flip_signs=False, permute=True),
#         "D": partial(transform_weights_biases, flip_signs=True, permute=False),
#     }
    
#     if name not in transform_functions:
#         raise ValueError(f"Invalid transform function: {name}")
    
#     return transform_functions[name]

def apply_equivariant_transformations(
    weights: Tuple,
    biases: Tuple,
    permute: bool = False,
    scale: bool = False,
    scale_type: str = 'relu',
    scale_std: float = 0.5
) -> (List[torch.Tensor], List[torch.Tensor]):
    """
    Applies random scaling and/or permutation group actions to a CNN's weights
    and biases, producing a functionally equivalent network for data augmentation.

    The function assumes a specific CNN architecture:
    - 3 Convolutional layers followed by a final Linear layer.
    - Channel progression: 1 -> 16 -> 16 -> 16 -> 10.
    - An activation function (ReLU or Tanh) follows each convolutional layer.

    Args:
        weights (Tuple[torch.Tensor, ...]): A tuple of weight tensors for the CNN.
        biases (Tuple[torch.Tensor, ...]): A tuple of bias tensors for the CNN.
        accuracy (float): The accuracy score associated with the network, passed through.
        permute (bool): If True, applies a random permutation to the output channels
                        of each hidden layer.
        scale (bool): If True, applies scaling to the output channels of each
                      hidden layer.
        scale_type (str): The type of scaling to apply, corresponding to the
                          activation function. Must be 'relu' or 'tanh'.
                          - 'relu': Multiplies by random positive numbers.
                          - 'tanh': Multiplies by +1 or -1 (sign-flipping).
        scale_std (float): For 'relu' scaling, this is the standard deviation
                           of the log-normal distribution for scaling factors.

    Returns:
        CNNBatch: A NamedTuple containing the new (augmented) weights, biases,
                  and the original accuracy score.
    """
    # --- 1. Input Validation and Setup ---
    if not permute and not scale:
        raise ValueError("At least one of 'permute' or 'scale' must be True.")

    if scale and scale_type not in {'relu', 'tanh'}:
        raise ValueError(f"Invalid scale_type '{scale_type}'. Must be 'relu' or 'tanh'.")

    num_layers = 4
    if len(weights) != num_layers or len(biases) != num_layers:
        raise ValueError(
            f"Expected {num_layers} weight/bias tensors, got {len(weights)}/{len(biases)}."
        )

    # Create copies to avoid modifying the original tensors in-place
    new_weights = [w.clone() for w in weights]
    new_biases = [b.clone() for b in biases]

    device = new_weights[0].device
    dtype = new_weights[0].dtype
    num_hidden_layers = 3

    # --- 2. Generate Transformations for Each Hidden Layer ---
    transformations = []
    for i in range(num_hidden_layers):
        num_channels = new_weights[i].shape[0]

        # Generate permutation vector (or identity if permute=False)
        if permute:
            p = torch.randperm(num_channels, device=device)
        else:
            p = torch.arange(num_channels, device=device)

        # Generate scaling vector (or identity if scale=False)
        if scale:
            if scale_type == 'relu':
                # log_min = 0.0
                # log_max = 5.0
                # exponents = torch.linspace(log_min, log_max, num_channels, device=device, dtype=dtype)
                
                # # Raise 10 to the power of these exponents.
                # s = 10.0 ** exponents
                
                # # Shuffle for randomness
                # s = s[torch.randperm(num_channels, device=device)]
                # Positive scaling factors for ReLU
                s = torch.exp(torch.randn(num_channels, device=device, dtype=dtype) * scale_std)
            else: # scale_type == 'tanh'
                # Sign flips (+1 or -1) for Tanh
                s = torch.randint(0, 2, (num_channels,), device=device, dtype=dtype) * 2 - 1
                s = s.to(dtype) # Ensure dtype consistency
        else:
            s = torch.ones(num_channels, device=device, dtype=dtype)

        transformations.append({'scale': s, 'perm': p})

    # --- 3. Apply Transformations Layer by Layer ---

    # Layer 0: First convolutional layer (Conv1)
    s_out, p_out = transformations[0]['scale'], transformations[0]['perm']
    # Apply this layer's output transformation
    new_weights[0].mul_(s_out.view(-1, 1, 1, 1))
    new_biases[0].mul_(s_out)
    new_weights[0] = new_weights[0][p_out]
    new_biases[0] = new_biases[0][p_out]

    # Layers 1 and 2: Intermediate convolutional layers (Conv2, Conv3)
    for i in range(1, num_hidden_layers):
        s_in, p_in = transformations[i-1]['scale'], transformations[i-1]['perm']
        s_out, p_out = transformations[i]['scale'], transformations[i]['perm']

        # A. Compensate for the previous layer's transformation on input channels (dim 1)
        # The inverse transformation is (P*S)^-1 = S^-1 * P^-1.
        # We first compensate for scaling, then for permutation.
        new_weights[i].div_(s_in.view(1, -1, 1, 1))
        new_weights[i] = new_weights[i][:, p_in, :, :]

        # B. Apply this layer's output transformation
        new_weights[i].mul_(s_out.view(-1, 1, 1, 1))
        new_biases[i].mul_(s_out)
        new_weights[i] = new_weights[i][p_out]
        new_biases[i] = new_biases[i][p_out]

    # Layer 3: Final linear layer
    s_in, p_in = transformations[-1]['scale'], transformations[-1]['perm']
    # Compensate for the last hidden layer's transformation on input features (dim 1)
    new_weights[-1].div_(s_in.view(1, -1))
    new_weights[-1] = new_weights[-1][:, p_in]
    # The final biases are not transformed.

    return new_weights, new_biases

@torch.no_grad()
def test_cnn_orbit(
    dataset,
    model_config: dict,
    device: torch.device,
    tolerance: float = 1e-3
):
    """
    Tests the unbatched augmentation function by creating many augmented versions
    of a base CNN, batching them, and verifying their functional equivalence.

    Args:
        base_cnn (CNNBatch): A CNNBatch for a SINGLE base model (unbatched weights).
        model_config (dict): Configuration dictionary for evaluation.
        device (torch.device): The device to run evaluation on.
        num_augmentations (int): Number of augmented versions to generate.
        tolerance (float): Maximum allowable difference in accuracy.
    """
    # print("--- Starting CNN Orbit Sanity Check (Augment then Batch Workflow) ---")

    # # --- 1. Generate many augmented versions of the base model ---
    # augmented_cnn_list = []
    
    # # Add the original model as the first element for a baseline
    # # squeeze the batch dimension to get a single model
    # base_cnn = CNNBatch(
    #     weights=tuple(w.squeeze(0) for w in base_cnn.weights),
    #     biases=tuple(b.squeeze(0) for b in base_cnn.biases),
    #     y=base_cnn.y
    # )
    # print("Accuracy of the base model:", base_cnn.y.item())
    # augmented_cnn_list.append(base_cnn)
    
    # print(f"Generating {num_augmentations} augmented versions of the base model...")
    # scale_type = model_config['data']['activation_function'] # Assumes 'relu' or 'tanh'

    # transformations_mapping = {"PD":{'permute': True, 'scale': True},
    #                             "P": {'permute': True, 'scale': False},
    #                             "D": {'permute': False, 'scale': True}}
    # # Get the specific transformation function based on the config
    # transformations = transformations_mapping[args.transform_type]
    # for _ in tqdm(range(num_augmentations), desc="Generating Augmentations"):
    #     # Call your specific unbatched function
    #     new_weights_list, new_biases_list = apply_equivariant_transformations(
    #         weights=base_cnn.weights,
    #         biases=base_cnn.biases,
    #         permute=transformations['permute'],
    #         scale=transformations['scale'],
    #         scale_type=scale_type,
    #         scale_std=5
    #     )
    #     # Wrap the returned lists into a CNNBatch object
    #     aug_cnn = CNNBatch(
    #         weights=tuple(new_weights_list),
    #         biases=tuple(new_biases_list),
    #         y=base_cnn.y
    #     )
    #     augmented_cnn_list.append(aug_cnn)
        
    # # --- 2. Collate the list of single models into one Batched CNNBatch ---
    # print("\nCollating list of single models into one large batch...")
    # # This helper function remains the same and is crucial

    # print(f"Number of models in the batch: {len(augmented_cnn_list)}")
    # # print all the layers shapes 
    # print("Shapes of all layers in the first model:")
    # for i, (w, b) in enumerate(zip(augmented_cnn_list[0].weights, augmented_cnn_list[0].biases)):
    #     print(f"Layer {i}: weights shape: {w.shape}, biases shape: {b.shape}")

    # final_batched_cnn = collate_cnn_batches(augmented_cnn_list)
    # print(f"Final batch contains {len(final_batched_cnn.y)} models.")
    # print(f"Shape of first batched weight tensor: {final_batched_cnn.weights[0].shape}")

    # --- 3. Prepare for evaluation ---
    batched_params_to_test = (
        tuple(w.to(device) for w in dataset.weights),
        tuple(b.to(device) for b in dataset.biases)
    )

    # --- 4. Evaluate the entire batch of models on CIFAR-10 ---
    cifar_loader = get_cifar10_test_loader(batch_size=256)
    num_models = len(dataset.y)
    correct_preds = torch.zeros(num_models, device=device)
    total_samples = len(cifar_loader.dataset)

    print("\nEvaluating the entire batch of models on CIFAR-10...")
    for images, labels in tqdm(cifar_loader, desc="Testing Full Batch"):
        images, labels = images.to(device), labels.to(device)
        
        logits_batch = get_logits_for_batch(batched_params_to_test, images, model_config)
        preds_batch = torch.argmax(logits_batch, dim=2)
        correct_preds += (preds_batch == labels.unsqueeze(0)).sum(dim=1)
        
    accuracies = correct_preds / total_samples

    # --- 5. Analyze and Assert Results ---
    std_dev = torch.std(accuracies)
    
    print("\n--- Sanity Check Results ---")
    print(f"Accuracy of original model (model 0): {accuracies[0].item():.6f}")
    print(f"Mean accuracy of all {num_models} models: {torch.mean(accuracies).item():.6f}")
    print(f"Standard deviation of accuracies: {std_dev.item():.4e}")

    if std_dev > tolerance:
        raise AssertionError(
            f"Functional equivalence test FAILED! Accuracies are not identical. "
            f"Std dev: {std_dev.item()} > tolerance: {tolerance}"
        )
    
    print(f"✅ Sanity check passed. All {num_models} models in the batch have the same accuracy.")

def collate_cnn_batches(cnn_list: List[CNNBatch]) -> CNNBatch:
    # ... (Implementation from previous answer)
    if not cnn_list:
        raise ValueError("Cannot collate an empty list of CNNs.")
    num_layers = len(cnn_list[0].weights)
    print(f"Collating {len(cnn_list)} CNNs with {num_layers} layers each.")
    batched_weights = tuple(torch.stack([cnn.weights[i] for cnn in cnn_list]) for i in range(num_layers))
    batched_biases = tuple(torch.stack([cnn.biases[i] for cnn in cnn_list]) for i in range(num_layers))
    batched_y = torch.tensor([cnn.y for cnn in cnn_list])
    return CNNBatch(weights=batched_weights, biases=batched_biases, y=batched_y)


def generate_orbit_dataset(
    output_dir: str,
    device: torch.device,
    dataset_size: int = 2 ** 12,
    transform_type: str = "PD",
    config: str = "./configs/cifar10_rec/scalegmn_relu.yml",
) -> None:
    """
    Compute the loss matrix for the given dataset using the CNN model.

    Args:
        output_dir (str): Directory to save the augmented dataset.
        base_cnn (CNNBatch): The original CNN model.
        device (torch.device): Device to use for computation (CPU or GPU).
        dataset_size (int): Number of augmented CNNs to generate.

    Returns:
        None
    """
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ensure deterministic behavior

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load arbitrary CNN
    # read config file
    effective_conf = yaml.safe_load(open(conf))
    set_seed(effective_conf["train_args"]["seed"])  # Use effective_conf
    # =============================================================================================
    #   PICK ONE CNN FROM THE DATASET TO AUGMENT
    # =============================================================================================
    equiv_on_hidden = mask_hidden(effective_conf)
    get_first_layer_mask = mask_input(effective_conf)
    
    test_set = dataset(effective_conf['data'],
                    split='test',
                    debug=False,
                    interpolation=True,
                    direction=effective_conf['scalegmn_args']['direction'],
                    equiv_on_hidden=equiv_on_hidden,
                    get_first_layer_mask=get_first_layer_mask)
    
    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=effective_conf["batch_size"],
        shuffle=False,
        num_workers=effective_conf["num_workers"],
        pin_memory=True,
    )

    # Get the one CNN (one batch but for interpolation mode it is only one sample)
    graph_batch, original_params_batch = next(iter(test_loader))
   
    graph_batch = graph_batch.to(device)
    original_params_batch = original_params_batch.to(device)

    base_cnn = CNNBatch(
       weights=tuple(w.squeeze(0) for w in original_params_batch.weights),
       biases=tuple(b.squeeze(0) for b in original_params_batch.biases),
       y=original_params_batch.y
    )

    transformations_mapping = {"PD":{'permute': True, 'scale': True},
                                "P": {'permute': True, 'scale': False},
                                "D": {'permute': False, 'scale': True}}
    # Get the specific transformation function based on the config
    transformations = transformations_mapping[transform_type]

    scale_type = effective_conf['data']['activation_function']

    dataset = []
    for cnn in tqdm(range(dataset_size), desc="Generating augmented dataset"):
        # Call your specific unbatched function
        new_weights_list, new_biases_list = apply_equivariant_transformations(
            weights=base_cnn.weights,
            biases=base_cnn.biases,
            permute=transformations['permute'],
            scale=transformations['scale'],
            scale_type=scale_type,
            scale_std=5
        )
        # Wrap the returned lists into a CNNBatch object
        aug_cnn = CNNBatch(
            weights=tuple(new_weights_list),
            biases=tuple(new_biases_list),
            y=base_cnn.y
        )
        dataset.append(aug_cnn)

        
    # --- 2. Collate the list of single models into one Batched CNNBatch ---
    print("\nCollating list of single models into one large batch...")
    print(f"Number of models in the batch: {len(dataset)}")
    # print all the layers shapes 
    print("Shapes of all layers in the first model:")
    for i, (w, b) in enumerate(zip(dataset[0].weights, dataset[0].biases)):
        print(f"Layer {i}: weights shape: {w.shape}, biases shape: {b.shape}")

    dataset = collate_cnn_batches(dataset)

    # Test the augmented dataset
    # If it fails the test, it will raise an AssertionError and stop the execution
    test_orbit_dataset(dataset, effective_conf, device, tolerance=1e-5)

    os.makedirs(output_dir, exist_ok=True)
    
    # Define the single file path for the entire dataset
    output_path = os.path.join(output_dir, "orbit_dataset.pt")
    
    print(f"\nSaving the entire batched orbit dataset to a single file: {output_path}")

    # Save the entire batched_dataset object
    torch.save(dataset, output_path)

    print(f"\n✅ Successfully saved {dataset_size} orbit samples to {output_path}.")
    print("To load the dataset, use: `loaded_dataset = torch.load('path/to/orbit_dataset.pt')`")




def delete_orbit_dataset(output_dir: str) -> None:
    """
    Delete the orbit dataset directory.

    Args:
        output_dir (str): Directory to delete.

    Returns:
        None
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted the directory: {output_dir}")
    else:
        print(f"The directory {output_dir} does not exist.")




# if __name__ == "__main__":
#     torch.set_float32_matmul_precision("high")
#     args = get_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # read config file
#     conf = yaml.safe_load(open(args.conf))
#     conf = overwrite_conf(conf, vars(args))


#     # If not using wandb, the original conf is the effective config
#     effective_conf = conf
#     run = None  # No wandb run object
 

#     set_seed(effective_conf["train_args"]["seed"])  # Use effective_conf

#     assert_symms(effective_conf)

#     # =============================================================================================
#     #   SETUP THE GRAPH DATASET FROM CNN AND DATALOADER
#     # =============================================================================================
#     if args.debug:
#         equiv_on_hidden = mask_hidden(effective_conf)
#         get_first_layer_mask = mask_input(effective_conf)
#         test_set = dataset(effective_conf['data'],
#                         split='test',
#                         debug=args.debug,
#                         direction=effective_conf['scalegmn_args']['direction'],
#                         equiv_on_hidden=equiv_on_hidden,
#                         get_first_layer_mask=get_first_layer_mask)

#     print(f'Len test set: {len(test_set)}')

#     # Define the common generator for the 2 dataloaders (original CNN and graph)
#     generator = torch.Generator().manual_seed(effective_conf["train_args"]["seed"])  # Use effective_conf

#     test_loader = torch_geometric.loader.DataLoader(
#         dataset=test_set,
#         batch_size=effective_conf["batch_size"],
#         shuffle=False,
#         num_workers=effective_conf["num_workers"],
#         pin_memory=True,
#     )

#     cifar10_test_loader = get_cifar10_test_loader(
#         batch_size=effective_conf["batch_size"],
#         num_workers=effective_conf["num_workers"],
#     )
#     # Get the one CNN (one batch but for debug mode it is only one sample)
#     graph_batch, original_params_batch = next(iter(test_loader))
   
#     graph_batch = graph_batch.to(device)
#     original_params_batch = original_params_batch.to(device)
#     gt_test_acc = graph_batch.y.to(device)

#     test_cnn_orbit(
#         base_cnn=original_params_batch,
#         model_config=effective_conf,
#         device=device,
#         num_augmentations=50,
#         tolerance=1e-2
#     )


    




    

