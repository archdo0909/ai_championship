from .champ import LGADDataset

def load_dataset(dataset_name, data_path, random_state=None):

    implemented_datasets = ('ai_champ')

    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'ai_champ':
        dataset = LGADDataset(root=data_path, dataset_name=dataset_name)
    
    return dataset