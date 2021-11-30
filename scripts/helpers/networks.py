import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_trainable_parameters(self):
        parameters = 0

        for params in list(self.parameters()):
            elements = 1

            for size in list(params.size()):
                elements *= size

            parameters += elements

        return parameters

    @property
    def device(self):
        return next(self.parameters()).device


class MultiLayerPerceptron(BaseNetwork):
    def __init__(self, in_dim, out_dim, hidden_layers, activation=nn.Tanh, dropout=0.):
        super(MultiLayerPerceptron, self).__init__()
        assert type(hidden_layers) == list

        # Network Parameters
        self._in_dim = in_dim
        self._out_dim = out_dim

        self._layers = hidden_layers
        self._layers.insert(0, self._in_dim)
        self._layers.insert(len(self._layers), self._out_dim)

        self._num_layers = len(self._layers)
        self._activation = activation()
        self._dropout = nn.Dropout(p=dropout)

        # Network Layers
        self._fully_connected_layers = nn.ModuleList(
            [nn.Linear(self._layers[layer], self._layers[layer + 1]) for layer in range(self._num_layers - 2)]
        )

        self._output_layer = nn.Linear(self._layers[-2], self._layers[-1])

        # Latent State
        self._latent_tensor = None

    def forward(self, t):
        assert isinstance(t, torch.Tensor)

        for layer in self._fully_connected_layers:
            t = self._dropout(self._activation(layer(t)))

        self._latent_tensor = t
        return self._output_layer(t)

    @property
    def latent_tensor(self):
        return self._latent_tensor

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def layers(self):
        return self._layers


class GatedRecurrentUnit(BaseNetwork):
    def __init__(self, in_dim, hidden_state_dim):
        super(GatedRecurrentUnit, self).__init__()

        self._gates_input = nn.Linear(in_dim, hidden_state_dim * 3)
        self._gates_hidden = nn.Linear(hidden_state_dim, hidden_state_dim * 3)

    def forward(self, x, h):
        assert isinstance(x, torch.Tensor) and isinstance(h, torch.Tensor)

        r_input, z_input, n_input = self._gates_input(x).chunk(3, 1)
        r_hidden, z_hidden, n_hidden = self._gates_hidden(h).chunk(3, 1)

        r = nn.Sigmoid()(r_input + r_hidden)
        z = nn.Sigmoid()(z_input + z_hidden)
        n = nn.Tanh()(n_input + (r * n_hidden))

        return n + (z * (h - n))

    def _apply(self, fn):
        super(GatedRecurrentUnit, self)._apply(fn)

        try:
            self._gates_input = fn(self._gates_input)
            self._gates_hidden = fn(self._gates_hidden)
        except AttributeError as e:
            print('Warning:', e)

        return self
