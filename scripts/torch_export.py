import os

import torch
import torch.nn as nn

from helpers.utility import save_parameters
from helpers.networks import BaseNetwork, MultiLayerPerceptron, GatedRecurrentUnit


class ReferenceNetwork(BaseNetwork):
    def __init__(self, in_dim=2, hidden_state_dim=8, dropout=0.):
        super(ReferenceNetwork, self).__init__()
        self._recurrent_block = GatedRecurrentUnit(in_dim=in_dim, hidden_state_dim=hidden_state_dim)
        self._feed_forward_block = MultiLayerPerceptron(in_dim=in_dim + hidden_state_dim,
                                                        hidden_layers=[hidden_state_dim,
                                                                       hidden_state_dim],
                                                        out_dim=1, activation=nn.LeakyReLU, dropout=dropout)

        self._hidden_state_dim = hidden_state_dim
        self._hidden_state = torch.zeros((0, self._hidden_state_dim))
        self._hidden_state = self._hidden_state.to(next(self.parameters()).device)

    def forward(self, x):
        self._hidden_state = self._recurrent_block(x, self._hidden_state)
        return self._feed_forward_block(torch.cat([x, self._hidden_state], axis=1))

    def reset(self, batch_size=0, randomize=False):
        if randomize:
            self._hidden_state = torch.randn((batch_size, self._hidden_state_dim)) * 0.1
        else:
            self._hidden_state = torch.zeros((batch_size, self._hidden_state_dim))

        self._hidden_state = self._hidden_state.to(next(self.parameters()).device)

    @property
    def gru(self):
        return self._recurrent_block

    @property
    def mlp(self):
        return self._feed_forward_block


def main():
    # Create network object
    network = ReferenceNetwork(in_dim=2, hidden_state_dim=8, dropout=0.4)

    # Save torch parameters
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exported_parameters')
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(network.state_dict(), model_save_dir + '/state_dict.pt')
    torch.save(network, model_save_dir + '/model.pth')

    # Save parameters as .txt that can be loaded into the C++ networks
    save_parameters(network.gru, model_save_dir + '/gru_parameters.txt')
    save_parameters(network.mlp, model_save_dir + '/mlp_parameters.txt')


if __name__ == '__main__':
    main()
