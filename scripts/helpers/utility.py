import numpy as np


def save_parameters(network, save_path):
    network.eval()

    model_parameters = list(network.state_dict().keys())
    model_parameters = np.concatenate(
        [network.state_dict()[key].cpu().numpy().transpose().reshape(-1) for key in model_parameters])

    np.savetxt(save_path, model_parameters.reshape((1, -1)), delimiter=', ', newline='\n', fmt='%1.10f')

    print('\nSaved model parameters in the following order:')
    for parameter_key in list(network.state_dict().keys()):
        print('   ', parameter_key, '| Dimension:', network.state_dict()[parameter_key].shape)
