import torch
import torch_geometric
import yaml
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms as transforms
from src.data import dataset
from tqdm import tqdm
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.scalegmn.autoencoder import get_autoencoder
from src.scalegmn.cnn import unflatten_params_batch, get_logits_for_batch
from src.utils.loss import select_criterion, KnowledgeDistillationLoss
from src.utils.optim import setup_optimization
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed, mask_input, mask_hidden, count_named_parameters
from src.scalegmn.utils import get_cifar10_test_loader
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import wandb
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


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
            if isinstance(criterion, KnowledgeDistillationLoss):  
                current_elements = B_models * B_images
            elif isinstance(criterion, nn.MSELoss):
                current_elements = B_models * B_images * N_classes
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

def main():

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
    if effective_conf['debug']:
        train_set = dataset(effective_conf['data'],
                            split='train',
                            debug=effective_conf["debug"],
                            direction=effective_conf['scalegmn_args']['direction'],
                            equiv_on_hidden=equiv_on_hidden,
                            get_first_layer_mask=get_first_layer_mask)
        val_set = dataset(effective_conf['data'],
                        split='train',
                        debug=effective_conf["debug"],
                        direction=effective_conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)

        test_set = dataset(effective_conf['data'],
                        split='train',
                        debug=effective_conf["debug"],
                        direction=effective_conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)
    else:
        train_set = dataset(effective_conf['data'],
                            split='train',
                            direction=effective_conf['scalegmn_args']['direction'],
                            equiv_on_hidden=equiv_on_hidden,
                            get_first_layer_mask=get_first_layer_mask)
        val_set = dataset(effective_conf['data'],
                        split='val',
                        direction=effective_conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)

        test_set = dataset(effective_conf['data'],
                        split='test',
                        direction=effective_conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)


    param_shapes = {}
    for _, row in train_set.layout.iterrows():
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
    
    if 'debug_subset_fraction' in effective_conf and effective_conf['debug_subset_fraction'] < 1.0:
        fraction = effective_conf['debug_subset_fraction']
        print(f"!!! Using a {fraction*100:.0f}% subset of the data for faster iteration !!!")

        # Subset the training set
        train_size = len(train_set)
        train_subset_indices = np.random.choice(range(train_size), size=int(train_size * fraction), replace=False)
        train_set = Subset(train_set, train_subset_indices)

        # Subset the validation set
        val_size = len(val_set)
        val_subset_indices = np.random.choice(range(val_size), size=int(val_size * fraction), replace=False)
        val_set = Subset(val_set, val_subset_indices)

        test_size = len(test_set)
        test_subset_indices = np.random.choice(range(test_size), size=int(test_size * fraction), replace=False)
        test_set = Subset(test_set, test_subset_indices)

    # We typically don't need to subset the test set for sweeps, but you could if needed.

    print(f'Len  train set: {len(train_set)}')
    print(f'Len val set: {len(val_set)}')
    print(f'Len test set: {len(test_set)}')

    # Define the common generator for the 2 dataloaders (original CNN and graph)
    generator = torch.Generator().manual_seed(effective_conf["train_args"]["seed"])  # Use effective_conf

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=effective_conf["batch_size"],
        shuffle=True,  
        generator=generator,
        num_workers=effective_conf["num_workers"],
        pin_memory=True,
    )
    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=effective_conf["batch_size"],
        shuffle=False,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=effective_conf["batch_size"],
        shuffle=False,
        num_workers=effective_conf["num_workers"],
        pin_memory=True,
    )

    cifar10_test_loader = get_cifar10_test_loader(
        batch_size=effective_conf["batch_size"],
        num_workers=effective_conf["num_workers"],
    )

    # =============================================================================================
    #   DEFINE MODEL
    # =============================================================================================
    original_set = train_set.dataset if isinstance(train_set, Subset) else train_set
    effective_conf['scalegmn_args']["layer_layout"] = original_set.get_layer_layout()
    # conf['scalegmn_args']['input_nn'] = 'conv'
    #net = ScaleGMN(conf['scalegmn_args'])
    net = get_autoencoder(
        model_args=effective_conf,
        autoencoder_type=effective_conf["train_args"]["reconstruction_type"],
    )  # Use effective_conf
    print(net)

    # Print the number of parameters in the decoder and the entire network
    decoder_params = sum(p.numel() for p in net.decoder.parameters() if p.requires_grad)
    net_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Number of parameters in the decoder: {decoder_params}")
    print(f"Number of parameters in the entire network: {net_params}")

    cnt_p = count_parameters(net=net)
    if effective_conf.get(
        "wandb", False
    ):  # Use effective_conf and check if wandb is enabled
        wandb.log(
            {
                "number of parameters": cnt_p,
                "decoder parameters": decoder_params,
                "total parameters": net_params,
            },
            step=0,
        )

    for p in net.parameters():
        p.requires_grad = True

    net = net.to(device)
    # =============================================================================================
    #   DEFINE LOSS
    # =============================================================================================
    loss_args = effective_conf["train_args"].get("loss_args", {})
    criterion = select_criterion(effective_conf['train_args']['loss'], loss_args)

    # =============================================================================================
    #   DEFINE OPTIMIZATION
    # =============================================================================================
    conf_opt = effective_conf['optimization']
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf_opt['optimizer_name'], optimizer_args=conf_opt['optimizer_args'], scheduler_args=conf_opt['scheduler_args'])
    # =============================================================================================
    # TRAINING LOOP
    # =============================================================================================
    step = 0
    epochs_no_improve = 0
    patience = effective_conf['train_args']['patience']
    best_val_tau = -float("inf")
    best_val_loss = float("inf")
    best_val_l1_err = float("inf")
    best_train_tau_TRAIN = -float("inf")
    best_test_results, best_val_results, best_train_results, best_train_results_TRAIN = None, None, None, None

    for epoch in range(effective_conf['train_args']['num_epochs']):
        net.train()
        len_dataloader = len(train_loader)
        # progress_bar = tqdm(zip(train_loader, train_loader_original), total=len(train_loader))
        progress_bar = tqdm(train_loader,total=len_dataloader)
        for i, (graph_batch, original_params_batch) in enumerate(progress_bar):
            step = epoch * len_dataloader + i
            graph_batch = graph_batch.to(device)
            original_params_batch = original_params_batch.to(device)
            gt_test_acc = graph_batch.y.to(device)
            optimizer.zero_grad()
            # Forward pass through the network to get the vector of model parameters
            recon_params_flat = net(graph_batch)
            # Create a tuple of two BATCHED! lists of the weights and biases from the flattened parameters
            recon_params = unflatten_params_batch(recon_params_flat, original_set.layout, param_shapes) 
            # Define the original params
            original_params = (original_params_batch.weights, original_params_batch.biases)
            # Accumulate loss over the entire CIFAR-10 test set
            batch_loss = torch.tensor(0.0, device=device)
            total_elements = 0
            # === INNER LOOP OVER BATCHES OF CIFAR IMAGES ===
            for j, (cifar_images, _) in enumerate(cifar10_test_loader):
                cifar_images = cifar_images.to(device)
                # Get logits for both batches of models using the current image batch
                with torch.no_grad():
                    original_logits = get_logits_for_batch(original_params, cifar_images, effective_conf)
        
                recon_logits = get_logits_for_batch(recon_params, cifar_images, effective_conf)
                # Prepare logits for KLDivLoss
                B_models, B_images, N_classes = recon_logits.shape
                # The loss function expects input of shape (Number of Samples, Classes) 
                # But we have (B_models, B_images, N_classes)
                recon_logits_flat = recon_logits.view(-1, N_classes)
                original_logits_flat = original_logits.view(-1, N_classes)
                if isinstance(criterion, KnowledgeDistillationLoss):  
                    current_elements = B_models * B_images
                elif isinstance(criterion, nn.MSELoss):
                    current_elements = B_models * B_images * N_classes
                # Calculate loss for this mini-batch of images and accumulate
                batch_loss += criterion(recon_logits_flat, original_logits_flat) * current_elements
                total_elements += current_elements

            # 4. Average the total loss over all models and all test samples for logging
            avg_loss = batch_loss / total_elements
            avg_loss.backward()
            
            progress_bar.set_description(f"Epoch {epoch} | Avg Loss: {avg_loss.item():.6f}")
            #pred_acc = F.sigmoid(net(inputs)).squeeze(-1)
            
            log = {}
            if effective_conf['optimization']['clip_grad']:
                log['grad_norm'] = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                                  effective_conf['optimization']['clip_grad_max_norm']).item()

            optimizer.step()

            if effective_conf["wandb"]:
                if step % 1 == 0:
                    log[f"train/{effective_conf['train_args']['loss']}"] = avg_loss 
                    #log["train/rsq"] = r2_score(gt_test_acc.cpu().numpy(), pred_acc.detach().cpu().numpy())

                wandb.log(log, step=step)

            if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
                scheduler[0].step()

        #############################################
        # VALIDATION
        #############################################
        if effective_conf["validate"]:
            print(f"\nValidation after epoch {epoch}:")
            eval_criterion = select_criterion(effective_conf['train_args']['loss'], {})

            # Define the arguments dict to pass to the evaluate function
            eval_args = {
                "net": net,
                "criterion": eval_criterion,
                "cifar10_test_loader": cifar10_test_loader,
                "device": device,
                "layout": original_set.layout,
                "param_shapes": param_shapes,
                "effective_conf": effective_conf,
            }
            # --- Evaluate on Validation Set ---
            val_loss_dict = evaluate(
                loader=val_loader,
                # graph_loader=val_loader, 
                # original_params_loader=val_loader_original,
                **eval_args
            )
            print(f"Epoch {epoch}, val L1 err: {val_loss_dict['avg_err']:.9f}, val loss: {val_loss_dict['avg_loss']:.9f}, val Rsq: {val_loss_dict['rsq']:.9f}, val tau: {val_loss_dict['tau']}")
            # --- Evaluate on Test Set ---
            test_loss_dict = evaluate(
                loader=test_loader,
                # graph_loader=test_loader, 
                # original_params_loader=test_loader_original,
                **eval_args
            )
            # Print the test results
            print(f"Epoch {epoch}, test L1 err: {test_loss_dict['avg_err']:.9f}, test loss: {test_loss_dict['avg_loss']:.9f}, test Rsq: {test_loss_dict['rsq']:.9f}, test tau: {test_loss_dict['tau']}")
     
            # --- Evaluate on Train Set (optional, can be slow) ---
            # train_loss_dict = evaluate(
            #     loader=train_loader,  
            #     **eval_args
            # )

            best_val_criteria = val_loss_dict['avg_err'] <= best_val_l1_err
            if best_val_criteria:
                best_val_loss = val_loss_dict['avg_loss']
                best_val_l1_err = val_loss_dict['avg_err']
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                #best_train_results = train_loss_dict

            # Save the model
            if effective_conf["save_model"]["save_model"]:  # Use effective_conf
                if (
                    best_val_criteria and effective_conf["save_model"]["save_best"]
                ):  # Use effective_conf
                    save_path = (
                        effective_conf["save_model"]["save_dir"]
                        + "/"
                        + effective_conf["save_model"]["save_name"]
                    )  # Use effective_conf
                    if not os.path.exists(
                        effective_conf["save_model"]["save_dir"]
                    ):  # Use effective_conf
                        os.makedirs(
                            effective_conf["save_model"]["save_dir"], exist_ok=True
                        )  # Use effective_conf
                    torch.save(net.state_dict(), save_path)

            # best_train_criteria = train_loss_dict['tau'] >= best_train_tau_TRAIN
            # if best_train_criteria:
            #     best_train_tau_TRAIN = train_loss_dict['tau']
            #     best_train_results_TRAIN = train_loss_dict

            if effective_conf["wandb"]:
                plt.clf()
                plot = plt.scatter(val_loss_dict['actual'], val_loss_dict['pred'])
                plt.xlabel("Actual model accuracy")
                plt.ylabel("Predicted model accuracy")
                wandb.log({
                    # "train/l1_err": train_loss_dict['avg_err'],
                    # "train/loss": train_loss_dict['avg_loss'],
                    # "train/rsq": train_loss_dict['rsq'],
                    # "train/kendall_tau": train_loss_dict['tau'],
                    # "train/best_rsq": best_train_results['rsq'] if best_train_results is not None else None,
                    # "train/best_tau": best_train_results['tau'] if best_train_results is not None else None,
                    # "train/best_rsq_TRAIN_based": best_train_results_TRAIN['rsq'] if best_train_results_TRAIN is not None else None,
                    # "train/best_tau_TRAIN_based": best_train_results_TRAIN['tau'] if best_train_results_TRAIN is not None else None,
                    "val/l1_err": val_loss_dict['avg_err'],
                    "val/loss": val_loss_dict['avg_loss'],
                    "val/rsq": val_loss_dict['rsq'],
                    "val/scatter": wandb.Image(plot),
                    "val/kendall_tau": val_loss_dict['tau'],
                    "val/best_rsq": best_val_results['rsq'] if best_val_results is not None else None,
                    "val/best_tau": best_val_results['tau'] if best_val_results is not None else None,
                    "val/best_l1_err": best_val_results['avg_err'] if best_val_results is not None else None,
                    # test
                    "test/l1_err": test_loss_dict['avg_err'],
                    "test/loss": test_loss_dict['avg_loss'],
                    "test/rsq": test_loss_dict['rsq'],
                    "test/kendall_tau": test_loss_dict['tau'],
                    "test/best_rsq": best_test_results['rsq'] if best_test_results is not None else None,
                    "test/best_tau": best_test_results['tau'] if best_test_results is not None else None,
                    "test/best_l1_err": best_test_results['avg_err'] if best_test_results is not None else None,
                    "epoch": epoch
                }, step=step)

            net.train() 

        # Save model every n epochs
        if (
            effective_conf["save_model"]["save_model"]
            and epoch % effective_conf["save_model"]["save_every"] == 0
        ):
            save_path = (
                effective_conf["save_model"]["save_dir"]
                + "/checkpoints/"
                + "epoch="
                + str(epoch)
                + "_"
                + effective_conf["save_model"]["save_name"]
            )  # Use effective_conf
            if not os.path.exists(
                effective_conf["save_model"]["save_dir"] + "/checkpoints/"
            ):  # Use effective_conf
                os.makedirs(
                    effective_conf["save_model"]["save_dir"] + "/checkpoints/",
                    exist_ok=True,
                )  # Use effective_conf
            torch.save(net.state_dict(), save_path)
        
        if epochs_no_improve >= patience and effective_conf["train_args"]["early_stopping"]:
            print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
            break

    if run:
        wandb.finish()  # Finish the W&B run if it was initialized
#  
# @torch.no_grad()
# def evaluate(net, loader, loss_fn, device):
#     net.eval()
#     pred, actual = [], []
#     err, losses = [], []
#     for batch in loader:
#         batch = batch.to(device)
#         gt_test_acc = batch.y.to(device)
#         inputs = batch.to(device)
#         pred_acc = F.sigmoid(net(inputs)).squeeze(-1)

#         err.append(torch.abs(pred_acc - gt_test_acc).mean().item())
#         losses.append(loss_fn(pred_acc, gt_test_acc).item())
#         pred.append(pred_acc.detach().cpu().numpy())
#         actual.append(gt_test_acc.cpu().numpy())

#     avg_err, avg_loss = np.mean(err), np.mean(losses)
#     actual, pred = np.concatenate(actual), np.concatenate(pred)
#     rsq = r2_score(actual, pred)
#     tau = kendalltau(actual, pred).correlation

#     return {
#         "avg_err": avg_err,
#         "avg_loss": avg_loss,
#         "rsq": rsq,
#         "tau": tau,
#         "actual": actual,
#         "pred": pred
#     }



if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    main()