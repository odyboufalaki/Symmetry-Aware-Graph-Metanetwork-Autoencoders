> This work extends previous research and public codebases, please refer to the [credit attribution](#credit-attribution) section.


# Symmetry-Aware Graph Metanetwork Autoencoders: Model Merging through Parameter Canonicalization
<!-- Link to the paper: [[Title of our paper](https://arxiv.org/)] -->
## Abstract

Neural network parameterizations exhibit inherent symmetries that yield multiple equivalent minima within the loss landscape. Scale Graph Metanetworks (ScaleGMNs) explicitly leverage these symmetries by proposing an architecture equivariant to both permutation and parameter scaling transformations. Previous work by Ainsworth et al.~(2023) addressed permutation symmetries through a computationally intensive combinatorial assignment problem, demonstrating that leveraging permutation symmetries alone can map networks into a shared loss basin. In this work, we extend their approach by also incorporating scaling symmetries, presenting an autoencoder framework utilizing ScaleGMNs as invariant encoders. Experimental results demonstrate that our method aligns Implicit Neural Representations (INRs) and Convolutional Neural Networks (CNNs) under both permutation and scaling symmetries without explicitly solving the assignment problem. This approach ensures that similar networks naturally converge within the same basin, facilitating model merging, i.e., smooth linear interpolation while avoiding regions of high loss.

## [Setup](#setup)
Create the virtual environment:
```bash
conda env create -n environment --file environment.yml
conda activate environment
```
Navigate to the root directory of the project and export the PYTHONPATH:
```bash
export PYTHONPATH=$PWD
```
Download and preprocess the data.
For the INR datasets we use the data provided by [DWS](https://github.com/AvivNavon/DWSNets), [MNIST-INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=mnist-inrs.zip) - ([Navon et al. 2023](https://arxiv.org/abs/2301.12780)):
```bash
# Create and export the data directory
mkdir data
export DATA_DIR=./data

# Download the MNIST-INRs data.
# Alternatively download manually, unzip and place in /data
wget "https://www.dropbox.com/sh/56pakaxe58z29mq/AABrctdu2U65jGYr2WQRzmMna/mnist-inrs.zip?dl=0" -O "$DATA_DIR/mnist-inrs.zip"
unzip -q "$DATA_DIR/mnist-inrs.zip" -d "$DATA_DIR"
rm "$DATA_DIR/mnist-inrs.zip"

# Generate the dataset splits
python src/utils/generate_data_splits.py --data_path $DATA_DIR/mnist-inrs --save_path $DATA_DIR/mnist-inrs

# Phase canonicalize
python src/phase_canonicalization/canonicalization.py --conf src/phase_canonicalization/mnist.yml
```
For the CNN datasets we use the CIFAR10 dataset provided by [Unterthiner et al, 2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy). The CIFAR10 data can be dowloaded from [here](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz). Extract the dataset in the directory `data/` and execute:

```bash
tar -xvf cifar10.tar.xz

wget https://github.com/AllanYangZhou/nfn/raw/refs/heads/main/experiments/predict_gen_data_splits/cifar10_split.csv -O data/cifar10/cifar10_split.csv
```

Download the MNIST dataset for the interpolation experiment:
```bash 
python script_utils/download_mnist.py
```

## [INR Experiments](#experiments)
In an identical manner to [1], for every experiment we provide the corresponding configuration file in the `config/` directory.
Each config contains the selected hyperparameters for the experiment, as well as the paths to the dataset.
To enable wandb logging, use the CLI argument `--wandb True`.
### Preliminary step: Train the Autoencoders
```bash
# Train the ScaleGMN autoencoder (wandb optional)
python train_autoencoder.py --conf configs/mnist_rec/scalegmn_autoencoder.yml --wandb True
```

```bash
# Train the Neural Graphs autoencoder
python src/neural_graphs/experiments/inr_classification/main.py model=pna data=mnist
```

Optionally train the autoencoder with the ablation of the scale canonicalization.

```bash
# Train the ScaleGMN autoencoder with scale ablation (wandb optional)
python train_autoencoder.py --conf configs/mnist_rec/scalegmn_autoencoder_ablation.yml --wandb True
```

Now, in order to generalize the commands from the experiments first define the saved model weights and config file as environment variables. For the ScaleGMN these are as follows:
```bash
export MODEL_CONFIG=configs/mnist_rec/scalegmn_autoencoder.yml
export MODEL_WEIGHTS=models/mnist_rec_scale/scalegmn_autoencoder/scalegmn_autoencoder_mnist_rec.pt
```

### Experiment 1: Visualizing the latent space
Once we have trained the autoencoders the first experiment is to visualize the latent space, for which we use `analysis/visualization(_ng).py`
To this end, we fit two dimensionality reduction methods: [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) and [uMAP](https://arxiv.org/pdf/1802.03426).
```bash
# Visualize the latent space for the ScaleGMN autoencoder
python analysis/visualization.py --conf $MODEL_CONFIG --ckpt $MODEL_WEIGHTS --outdir analysis/resources/visualization --seed 0
```

```bash
# Visualize the latent space for the Neural Graphs autoencoder
export MODEL_WEIGHTS=outputs/2025-05-11/16-21-51/5gzpb5lt/best_val.ckpt 
python analysis/visualization_ng.py --ckpt $MODEL_WEIGHTS --outdir analysis/resources/visualization --seed 0
```

### Experiment 2: Visualize the INR orbits
We now empirically test the effect of the encoders by generating an orbit dataset via the script `analysis/utils/orbit_dataset.py`.
We first define the type of group action to be applied to the weights and biases of an INR in order to generate the orbit dataset (a dataset of functionally equivalent INRs but that live in different basins).
These are the options for generating the orbits orbit:

    - P: Only permute
    - D: Only flip the signs (+1, -1)
    - PD: First flip signs then permute (both)

```bash
python analysis/utils/orbit_dataset.py --transform_type D
```

We now pass the orbit INRs through the different architectures using the script `analysis/plot_orbit(_ng).py`.
It is expected to see how ScaleGMN collapses the orbit for permutations and sign flippings, while Neural Graphs are only able to collapse the permuted orbit.
```bash
# Visualize the orbit for the ScaleGMN 
python analysis/plot_orbit.py --conf $MODEL_CONFIG --ckpt $MODEL_WEIGHTS --outdir analysis/resources/visualization

```

```bash
# Visualize the orbit for the Neural Graphs
python analysis/plot_orbit_ng.py --ckpt $MODEL_WEIGHTS --outdir analysis/resources/visualization

```

### Experiment 3: Interpolation experiment

This is the main experiment, in which we replicate the [Git Re-Basin](https://arxiv.org/pdf/2209.04836) approach for the autoencoder architecture.

We first define the type of transformation to define the orbit dataset.
```bash
export INTERPOLATION_TYPE=PD
```

The main functionality of the experiment is defined in the script `analysis/orbit_interpolation(_ng).py`.

This script takes care of the generation of the orbits according to the value of `INTERPOLATION_TYPE`.
The result of these commands will be two loss matrices storing the reconstruction loss (MSE with the original MNIST digit image) for different values of the interpolation parameter.
The first matrix will correspond to naive interpolation and the other to the reconstruction with autoencoder or, optionally for `analysis/orbit_interpolation.py`, the linear assignment.
For each run of a total of `num_runs`, a matrix of dimensions `[dataset_size, num_interpolations]` is created (`num_interpolations = 40` is enough level of detail to understand the continuous interpolation).
These matrices are concatenated to generate the final output `[dataset_size * num_runs, num_interpolations]`, which is then fed to `analysis/plot_interpolation.py`.


<!-- ```bash
# Interpolate using linear assignment
python analysis/orbit_interpolation.py --conf $MODEL_CONFIG --tmp_dir analysis/tmp_dir --dataset_size 512 --split test --seed 0 --num_runs 10 --perturbation 0.005 --linear_assignment $INTERPOLATION_TYPE --orbit_transformation $INTERPOLATION_TYPE --save_matrices
```

```bash
# Interpolate using ScaleGMN autoencoder
python analysis/orbit_interpolation.py --conf $MODEL_CONFIG --ckpt $MODEL_WEIGHTS --tmp_dir analysis/tmp_dir --dataset_size 512 --split test --seed 0 --num_runs 10 --perturbation 0.005 --orbit_transformation $INTERPOLATION_TYPE --save_matrices
``` -->

```bash
# Interpolate using ScaleGMN autoencoder, linear assignment and naive methods
python analysis/orbit_interpolation.py --conf $MODEL_CONFIG --ckpt $MODEL_WEIGHTS --tmp_dir analysis/tmp_dir --dataset_size 512 --split test --seed 0 --num_runs 10 --perturbation 0.005 --orbit_transformation $INTERPOLATION_TYPE --experiments naive scalegmn lap --save_matrices
```

```bash
# Interpolate using Neural Graphs autoencoder
python analysis/orbit_interpolation_ng.py --ckpt $MODEL_WEIGHTS --tmp_dir analysis/tmp_dir --dataset_size 512 --split test --seed 0 --num_runs 10 --perturbation 0.005 --orbit_transformation $INTERPOLATION_TYPE --save_matrices 
```

Finally generate the interpolation plots using the loss matrices.
The logic behind `analysis/plot_interpolation.py` is to average over all INR pairs and all runs to plot the expected interpolation curve for the different methods (Naive, ScaleGMN autoencoder, Neural Graphs autoencoder and Linear Assignment).
```bash
python analysis/plot_interpolation.py --matrix_dir analysis/resources/interpolation/matrices --output_dir analysis/resources/interpolation
```

## [CNN Experiments](#cnn_experiments)

As with the [INR experiments](#experiments), this section provides instructions for training CNN autoencoders and performing interpolation experiments.

### Preliminary Step: Train CNN Autoencoders

```bash
# Train the ScaleGMN CNN Autoencoders (wandb optional)

# For ReLU activation
python train_autoencoder_cnn.py --conf configs/cifar10_rec/scalegmn_relu.yml --wandb True 

# For Tanh activation
python train_autoencoder_cnn.py --conf configs/cifar10_rec/scalegmn_tanh.yml --wandb True
```

```bash
# Train the Neural Graphs CNN Autoencoders (wandb optional)

# For ReLU activation
python src/neural_graphs/experiments/cnn_generalization/main.py model=pna data=zoo_cifar_nfn_relu distributed=False

# For Tanh activation
python src/neural_graphs/experiments/cnn_generalization/main.py model=pna data=zoo_cifar_nfn_tanh distributed=False
```
### Interpolation For Multiple Pairs
This experiment generates interpolation data for multiple pairs of models. This data will then be used to plot the average curve along with the corresponding variance. The commands below will save the raw interpolation data to files within `analysis/resources/interpolation/cnn/`. Specifically, you'll find output files like `interpolation_results.pt` for ScaleGMN and `interpolation_results_ng.pt` for Neural Graphs.

First, run the interpolation for models trained with **ReLU** activation:
```bash
# ScaleGMN (ReLU) - Update 'checkpoint_path' with your actual path
python ./analysis/orbit_interpolation_cnn.py --conf ./configs/cifar10_rec/scalegmn_relu.yml --checkpoint_path ./models/cifar10_relu_rec/scalegmn_autoencoder/scalegmn_autoencoder_cifar_relu_rec.pt --train_args.seed 0 --interpolation_methods original reconstructed linear_assignment --multiple_pairs
# Neural Graphs (ReLU) - Update 'load_ckpt' with your actual path
python ./analysis/orbit_interpolation_cnn_ng.py batch_size=64 load_ckpt=./outputs/2025-07-17/19-29-20/7eluohps/best_val.ckpt interpolation_many_pairs=True model=pna data=zoo_cifar_nfn_relu distributed=False seed=0
```
After generating the interpolation data, you can plot the accuracy and loss curves using the following commands:
```bash
# Plot accuracies
python analysis/plot_interpolation_cnn.py --metric accuracies
# Plot losses
python analysis/plot_interpolation_cnn.py --metric losses
```

Now, run the interpolation for models trained with **Tanh** activation:

```bash
# ScaleGMN (Tanh) - Update 'checkpoint_path' with your actual path
python ./analysis/orbit_interpolation_cnn.py --conf ./configs/cifar10_rec/scalegmn_tanh.yml --checkpoint_path ./models/cifar10_tanh_rec/scalegmn_autoencoder/checkpoints/epoch=280_scalegmn_autoencoder_cifar_tanh_rec.pt --train_args.seed 0 --interpolation_methods original reconstructed linear_assignment --multiple_pairs

# Neural Graphs (Tanh) - Update 'load_ckpt' with your actual path
python ./analysis/orbit_interpolation_cnn_ng.py batch_size=64 load_ckpt=./outputs/2025-07-17/18-09-30/3avaj28g/best_val.ckpt interpolation_many_pairs=True model=pna data=zoo_cifar_nfn_tanh distributed=False seed=0
```
After generating the interpolation data, you can plot the accuracy and loss curves using the following commands: 
```bash
# Plot accuracies
python analysis/plot_interpolation_cnn.py --metric accuracies
# Plot losses
python analysis/plot_interpolation_cnn.py --metric losses
```

### Interpolation For Independent Pairs

This experiment focuses on visualizing the interpolation curve between a single pair of CNNs. First, generate the interpolation data for models with **ReLU** activation:

```bash
# ScaleGMN (ReLU)
python ./analysis/orbit_interpolation_cnn.py --conf ./configs/cifar10_rec/scalegmn_relu.yml --checkpoint_path ./models/cifar10_relu_rec/scalegmn_autoencoder/scalegmn_autoencoder_cifar_relu_rec.pt --train_args.seed 65 --interpolation_methods original reconstructed linear_assignment

# Neural Graphs (ReLU) 
python ./analysis/orbit_interpolation_cnn_ng.py batch_size=64 load_ckpt=./outputs/2025-07-17/19-29-20/7eluohps/best_val.ckpt interpolation=True model=pna data=zoo_cifar_nfn_relu distributed=False seed=65
```

After running the above commands to generate all the necessary interpolation data, use the following commands to plot the accuracy and loss curves:

```bash
# Plot accuracies
python analysis/plot_interpolation_cnn.py --metric accuracies
# Plot losses
python analysis/plot_interpolation_cnn.py --metric losses
```

In a similar manner, you can first generate and then plot the results for models with **Tanh** activation:

```bash
# ScaleGMN (Tanh)
python ./analysis/orbit_interpolation_cnn.py --conf ./configs/cifar10_rec/scalegmn_tanh.yml --checkpoint_path ./models/cifar10_tanh_rec/scalegmn_autoencoder/checkpoints/epoch=280_scalegmn_autoencoder_cifar_tanh_rec.pt --train_args.seed 7 --interpolation_methods original reconstructed linear_assignment

# Neural Graphs (Tanh) 
python ./analysis/orbit_interpolation_cnn_ng.py batch_size=64 load_ckpt=./outputs/2025-07-17/18-09-30/3avaj28g/best_val.ckpt interpolation=True model=pna data=zoo_cifar_nfn_tanh distributed=False seed=7

# Plot accuracies
python analysis/plot_interpolation_cnn.py --metric accuracies
# Plot losses
python analysis/plot_interpolation_cnn.py --metric losses
```


## [Credit attribution](#credit-attribution)
This repository and our research builds upon the work of I. Kalogeropoulos et al. [1] and M. Kofinas et al. [2] and their respective codebases.


[1] Kalogeropoulos, I. et al. (2024). *Scale Equivariant Graph Metanetworks.* Advances in Neural Information Processing Systems 37 (NeurIPS 2024). https://arxiv.org/pdf/2406.10685

[2] Kofinas, M. et al. (2024). *Graph Neural Networks for Learning Equivariant Representations of Neural Networks.* In 12th International Conference on Learning Representations (ICLR). https://arxiv.org/pdf/2403.12143.
