import torch
import torch.nn as nn
from abc import ABC
from ..data.base_datasets import Batch
from .models import ScaleGMN
from .inr import *
from .mlp import mlp

def create_batch_wb(
    params_flatten: list[torch.Tensor],
    *,
    in_features: int = 2,
    n_layers: int = 3,
    hidden_features: int = 32,
    out_features: int = 1,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Reconstruct the state dict of an INR model from a flat parameter vector.

    Args:
        params_flatten:  1D tensor containing all weights & biases for an INR with this arch.
        in_features:     INR.__init__ in_features
        n_layers:        INR.__init__ n_layers
        hidden_features: INR.__init__ hidden_features
        out_features:    INR.__init__ out_features
        fix_pe:          INR.__init__ fix_pe

    Returns:
        weights:         List of tensors containing the weights of the INR model.
        biases:          List of tensors containing the biases of the INR model.
    """
    # Instantiate fresh INR
    model = INR(
        in_features=in_features,
        n_layers=n_layers,
        hidden_features=hidden_features,
        out_features=out_features
    )

    # Break out its native params → shapes
    _, init_params = make_functional(model)
    _, shapes = params_to_tensor(init_params)

    batch_params_tuple = [tensor_to_params(params_flatten[k], shapes) for k in range(len(params_flatten))]

    batch_size = len(params_flatten)

    weights, biases = [], []
    for layer_k in range(len(shapes) // 2):
        weights_layer_k = torch.Tensor(shapes[2 * layer_k])
        weights_layer_k.unsqueeze(0)

        biases_layer_k = torch.Tensor(shapes[2 * layer_k + 1])
        biases_layer_k.unsqueeze(0)

        weights_layer_k = torch.cat(
            [batch_params_tuple[item][2 * layer_k].unsqueeze(0) for item in range(batch_size)],
            dim=0
        )

        biases_layer_k = torch.cat(
            [batch_params_tuple[item][2 * layer_k + 1].unsqueeze(0) for item in range(batch_size)],
            dim=0
        )

        weights.append(weights_layer_k.unsqueeze(-1))
        biases.append(biases_layer_k.unsqueeze(-1))

    return weights, biases


class Decoder(nn.Module, ABC):
    """
    Abstract base class for decoders in ScaleGMN.
    This class defines the interface for all decoders.
    """

    def __init__(self):
        self.net = None
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Tensor of shape [B, latent_dim]
        returns: Tensor of shape [B, output_dim]
        """
        return self.net(z)


class MLPDecoder(Decoder):
    """
    Generic MLP-based decoder for ScaleGMN embeddings.
    Mirrors the encoder to reconstruct the original signal/points.
    """
    def __init__(self, model_args, **kwargs):
        Decoder.__init__(self)
        self.input_dim = model_args['d_input']
        self.hidden_dims = model_args['d_hidden']
        self.num_layers = len(self.hidden_dims)
        self.data_layer_layout = model_args['data_layer_layout']
        self.output_dim = model_args['output_dim']
        self.activation = model_args['activation']

        self.net = mlp(
            in_features=self.input_dim,
            out_features=self.output_dim,
            d_k=self.hidden_dims,
            activation=self.activation,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Tensor of shape [B, latent_dim]
        returns: Tensor of shape [B, output_dim]
        """
        return self.net(z)


class Autoencoder(nn.Module, ABC):
    """
    Abstract base class for autoencoders in ScaleGMN.
    This class defines the interface for all autoencoders.
    """

    def __init__(self):
        self.encoder = None
        self.decoder = None
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, input_dim]
        returns: Tensor of shape [B, output_dim]
        """
        z = self.encoder(x)
        return self.decoder(z)


def get_autoencoder(model_args, autoencoder_type: str, **kwargs):
    _map = {
        "inr": _MLPAutoencoderINR,
        "pixels": _MLPAutoencoderPixels,
        "cnn": _MLPAutoencoderCNN,
    }
    autoencoder_class = _map.get(autoencoder_type, None)
    if autoencoder_class is None:
        raise ValueError(f"Unknown class name: {autoencoder_type}.")
    return autoencoder_class(model_args, **kwargs)


class _MLPAutoencoderINR(Autoencoder):
    """
    Generic MLP-based autoencoder for ScaleGMN embeddings.
    Mirrors the encoder to reconstruct the original signal/points.
    """
    def __init__(self, model_args, **kwargs):
        super().__init__()
        self.data_layer_layout = model_args["decoder_args"]['data_layer_layout']
        model_args["decoder_args"]["output_dim"] = sum([
            (self.data_layer_layout[i_layer] + 1) * self.data_layer_layout[i_layer + 1]
            for i_layer in range(len(self.data_layer_layout) - 1)
        ])
        self.encoder = ScaleGMN(model_args["scalegmn_args"], **kwargs)
        self.decoder = MLPDecoder(model_args["decoder_args"], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, input_dim]
        returns: Tensor of shape [B, output_dim]
        """
        z = self.encoder(x)
        return self.decoder(z)


class _MLPAutoencoderPixels(Autoencoder):
    """
    Generic MLP-based autoencoder for ScaleGMN embeddings.
    Mirrors the encoder to reconstruct the original signal/points.
    """
    def __init__(self, model_args, **kwargs):
        super().__init__()
        model_args["decoder_args"]["output_dim"] = torch.prod(torch.tensor(model_args["data"]["image_size"])).item()
        self.encoder = ScaleGMN(model_args["scalegmn_args"], **kwargs)
        self.decoder = MLPDecoder(model_args["decoder_args"], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, input_dim]
        returns: Tensor of shape [B, output_dim]
        """
        z = self.encoder(x)
        return self.decoder(z)


class _MLPAutoencoderCNN(Autoencoder):
    """
    Generic MLP-based autoencoder for ScaleGMN embeddings.
    Mirrors the encoder to reconstruct the original signal/points.
    """
    def __init__(self, model_args, **kwargs):
        super().__init__()
        self.data_layer_layout = model_args["decoder_args"]['data_layer_layout']
        # TODO : Maybe do this dynamically but all CNNs have the same output dim
        model_args["decoder_args"]["output_dim"] = 4970
        self.encoder = ScaleGMN(model_args["scalegmn_args"], **kwargs)
        self.decoder = MLPDecoder(model_args["decoder_args"], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, input_dim]
        returns: Tensor of shape [B, output_dim]
        """
        z = self.encoder(x)
        return self.decoder(z)


if __name__ == "__main__":
    # Example usage
    decoder = MLPDecoder(latent_dim=128, output_dim=784)
    z = torch.randn(32, 128)  # Batch of 32 samples
    output = decoder(z)
    print(output.shape)  # Should be [32, 784]