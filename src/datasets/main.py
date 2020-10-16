from .champ import LGADDataset

def load_dataset(dataset_name, data_path, normal_class, random_state=None):

    implemented_datasets = ('ai_champ')

    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'ai_champ':
        dataset = LGADDataset(root=data_path,
                                normal_class=normal_class,
                                )
    
    return dataset