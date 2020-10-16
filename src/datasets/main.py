from .champ import LGADDataset
from .mnist import MNIST_Dataset

def load_dataset(dataset_name, data_path, random_state=None):

    implemented_datasets = ('ai_champ', 'mnist')

    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'ai_champ':
        dataset = LGADDataset(root=data_path, 
                              dataset_name=dataset_name)
    
    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path,
                                normal_class=normal_class)
    
    return dataset