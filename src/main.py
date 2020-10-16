from datasets.main import load_dataset

def main(dataset_name, net_name, log_path, data_path, load_model):
    """

    Args:
        dataset_name ([type]): [description]
        net_name ([type]): [description]
        log_path ([type]): [description]
        data_path ([type]): [description]
        load_model ([type]): [description]
    """
    log_file = log_path + '/log.txt'

    dataset = load_dataset(dataset_name, data_path, 
                           random_state=100)