from .lg_LeNet import LG_LeNet, LG_LeNet_Autoencoder


def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('lg_LeNet')

    assert net_name in implemented_networks

    net = None

    if net_name == 'lg_LeNet':
        net = LG_LeNet()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('lg_LeNet')

    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'lg_LeNet':
        ae_net = LG_LeNet_Autoencoder()
    
    return ae_net