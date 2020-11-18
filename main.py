from torch.utils.data import DataLoader

from dataset import LGDataset
from dataset import SupervisedDataset
from dataset import DemonstrateDataset

from train import SampleTrainer
from test import SampleTester
from model import build_network

import torch
import logging


def main(data_path, data_name, xp_path, network, lr, n_epochs, batch_size, device, n_jobs_dataloader, stage_n_degc=None, train=True, supervised=False, demonstrate=False):
    """
        xp_path : 결과물 출력할 폴더의 절대 경로
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # TODO: Split dataset
    ds = SupervisedDataset if supervised else LGDataset
    if demonstrate:
        ds = DemonstrateDataset
        
    if train:
        train_set = ds(
            root=data_path,
            dataset_name=data_name,
            train=True,
            random_state=None,
            stage_n_degc=True
        )
        sample_train = SampleTrainer(
            lr=lr, n_epochs=n_epochs, batch_size=batch_size,
            device=device, n_jobs_dataloader=n_jobs_dataloader
        )
        network = build_network(network)

        weight_path = '/workspace/jinsung/crnn_random700_spectrogram-Copy1.pt'
        network_trained = sample_train.train(train_set, network, continuous=False, weight_path=weight_path)
        torch.save(network_trained.state_dict(), weight_path)

    else:
        test_set = ds(
            root=data_path,
            dataset_name=data_name,
            train=False,
            random_state=None
        )
        sample_test = SampleTester(
            lr=lr, n_epochs=n_epochs, batch_size=batch_size,
            device=device, n_jobs_dataloader=n_jobs_dataloader
        )
        network = build_network(network)
        network.load_state_dict(torch.load('/workspace/jinsung/resnet'))
        sample_test.test(test_set, network)


if __name__ == "__main__":
    # train
    main(
        data_path='/workspace/test1/201912_FLD165NBMA_vib_spectrum_modi_test1.txt',
        data_name='XYZ',
        xp_path='/workspace/ai_championship/log',
        network='CRNN',
        lr=0.001,
        n_epochs=10,
        batch_size=128,
        device='cuda',
        n_jobs_dataloader=4,
        stage_n_degc=True,
        supervised=True,
        demonstrate=True,
    )

    # test
    main(data_path='/workspace/peter/sampled/sampled.txt',
         data_name='sampled',
         xp_path='/workspace/ai_championship/log',
         network='CRNN',
         lr=0.001,
         n_epochs=1,
         batch_size=128,
         device='cuda',
         n_jobs_dataloader=4,
         stage_n_degc=True,
         train=False,
         supervised=True,
         demonstrate=True,
    )
