from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce
from typing import Any


# ============================================================ #
# ================== Activation Helper ======================= #
# ============================================================ #

_ACTIVATION_MAP = {
    "elu":     nn.ELU,
    "relu":    nn.ReLU,
    "tanh":    nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "selu":    nn.SELU,
}

def resolve_nn_activation(name: str) -> nn.Module:
    """
    Instantiate an activation module by name.

    Args:
        name (str): Activation function name (case-insensitive).
                    Supported: 'elu', 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'selu'.

    Returns:
        nn.Module: Instantiated activation module.

    Raises:
        ValueError: If the activation name is not recognised.
    """
    act_cls = _ACTIVATION_MAP.get(name.lower())
    if act_cls is None:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(_ACTIVATION_MAP.keys())}"
        )
    return act_cls()


def get_param(param: Any, idx: int) -> Any:
    """
    Retrieve a parameter by index, or return it directly if it is a scalar.

    Used by init_weights() to support both a single shared scale and
    per-layer scale tuples.

    Args:
        param (Any): A scalar value or a list/tuple of values.
        idx (int): Index to retrieve when param is a sequence.

    Returns:
        Any: The parameter value for the given index.
    """
    if isinstance(param, (tuple, list)):
        return param[idx]
    return param


# ============================================================ #
# ========================= MLP ============================== #
# ============================================================ #

class MLP(nn.Sequential):
    """
    Multi-layer perceptron (MLP) built as an nn.Sequential.

    Architecture:
        Linear → Activation → ... → Linear → Activation → Linear → (optional last activation)

    Conveniences:
        - Hidden dim of ``-1`` is replaced with ``input_dim`` (identity-width layer).
        - ``output_dim`` can be a tuple/list, in which case the final linear output
          is reshaped via nn.Unflatten.
        - An optional ``last_activation`` can be appended after the output layer.

    Args:
        input_dim (int): Number of input features.
        output_dim (int | tuple[int] | list[int]): Output dimension(s).
            Pass an int for a flat output, or a tuple/list for a shaped output.
        hidden_dims (tuple[int] | list[int]): Sizes of the hidden layers.
            Use -1 to inherit input_dim for that layer.
        activation (str): Activation function applied after every hidden layer.
            Default: 'elu'.
        last_activation (str | None): Optional activation after the output layer.
            None (default) leaves the output layer linear.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()

        # ===== Resolve activation functions ===== #
        activation_mod      = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None

        # ===== Replace -1 hidden dims with input_dim ===== #
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # ===== Build layers ===== #
        layers = []

        # Input → first hidden
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        # Hidden → hidden
        for i in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[i], hidden_dims_processed[i + 1]))
            layers.append(activation_mod)

        # Last hidden → output
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Optional last activation
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # ===== Register all layers ===== #
        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def init_weights(self, scales: float | tuple[float] = 1.0) -> None:
        """
        Initialise linear layer weights with orthogonal initialisation.

        Args:
            scales (float | tuple[float]): Orthogonal gain(s). Pass a single float
                to apply the same scale to every layer, or a tuple with one value
                per linear layer for per-layer control.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x (Tensor): Input tensor of shape (..., input_dim).

        Returns:
            Tensor: Output tensor of shape (..., output_dim).
        """
        for layer in self:
            x = layer(x)
        return x