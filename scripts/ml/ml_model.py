import torch
import torch.nn as nn


class WeatherMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list = None,
        output_dim: int = 1,
        activation=nn.ReLU,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)