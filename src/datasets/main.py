from .lg import LGADDataset
from .mnist import MNIST_Dataset
from .lg_svdd import LG_SVDD_Dataset


def svdd_load_dataset(dataset_name, data_path, normal_class, random_state):
    """Loads the dataset."""

    implemented_datasets = ('lg_svdd')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'lg_svdd':
        dataset = LG_SVDD_Dataset(root=data_path, dataset_name=dataset, normal_class=normal_class, random_state=random_state)

    return dataset

def load_dataset(dataset_name, data_path, normal_class, known_outlier_class=None, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('lg_train', 'lg_train_sample', 'mnist', 'augmented', 'lg_svdd')

    assert dataset_name in implemented_datasets

    dataset = None
    if dataset_name in ('lg_train'):
        dataset = LGADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
    
    if dataset_name in ('lg_train_sample'):
        dataset = LGADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
    
    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path,
                                normal_class=normal_class)
    
    if dataset_name in ('augmented'):
        dataset = LGADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == 'lg_svdd':
        dataset = LG_SVDD_Dataset(root=data_path, normal_class=normal_class)

    return dataset