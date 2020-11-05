from torch.utils.data import DataLoader
from dataset import LGDataset
from train import SampleTrainer
from model import build_network

import torch
import logging


def main(xp_path, network, lr, n_epochs, batch_size, device, n_jobs_dataloader):
    """
        xp_path : 결과물 출력할 폴더의 절대 경로
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file =  xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    train_set = LGDataset(root='/workspace/ai_championship/data',
                          dataset_name='lg_train',
                          train=True,
                          random_state=None)
    test_set = LGDataset(root='/workspace/ai_championship/data',
                         dataset_name='lg_train',
                         train=False,
                         random_state=None)

    sample_train = SampleTrainer(lr=lr, n_epochs=n_epochs, batch_size=batch_size,
                                 device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Register your network in model.py
    net = build_network(network)

    sample_train.train(train_set, net)

if __name__ == "__main__":

    main(xp_path='/workspace/ai_championship/log', network='Peter_CNN', lr=0.001, n_epochs=5, batch_size=2, device='cuda', n_jobs_dataloader=4)