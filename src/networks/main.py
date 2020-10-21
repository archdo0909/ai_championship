from .lg_LeNet import LG_LeNet, LG_LeNet_Autoencoder


def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('lg_LeNet', 'lg_')