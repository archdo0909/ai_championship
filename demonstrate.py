
import os
import time
import torch
import logging

import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import build_network


# (1) 다운로드한 데이터(폴더)에서 정상과 불량 데이터 분리
def split_data(data_dir):
    normal_data_fpath = os.path.join('/workspace/demon', 'normal.txt')
    abnormal_data_fpath = os.path.join('workspace/demon', 'abnormal.txt')

    if os.path.isfile(normal_data_fpath):
        print('normal.txt already exists, removing this file...')
        os.remove(normal_data_fpath)
    if os.path.isfile(abnormal_data_fpath):
        print('abnormal.txt already exists, removing this file...')
        os.remove(abnormal_data_fpath)

    testdata_dir = '/workspace/lg_train_test'
    data_fpaths = [os.path.join(testdata_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.txt')]
    for data_fpath in data_fpaths:
        with open(data_fpath, 'r') as f:
            # TODO: read line by line
            for line in f.readlines():
                print(line[:10])
                break

        with open(normal_data_fpath, 'a') as nf:
            nf.write()
        with open(abnormal_data_fpath, 'a') as af:
            af.write()


# (2) 위에서 저장된 데이터에 대하여 데이터 전처리 실행 및 저장
def preprocess_data(curr_X=None):
    if not curr_X:
        return None

    # curr_X: datetime, stage, temperature
    freqs = curr_X[3:-1]
    freqs_image = freqs.reshape(100, -1)

    stage = curr_X[1]
    temperature = curr_X[2]
    stage_channel = np.full(freqs_image.shape, stage, dtype=np.int8)
    temperature_channel = np.full(freqs_image.shape, temperature, dtype=np.float)

    time_str = str(int(curr_X[0]))
    month = int(time_str[3:5])
    month_channel = np.full(freqs_image.shape, month, dtype=np.int8)
    hour = int(time_str[7:9])
    hour_channel = np.full(freqs_image.shape, hour, dtype=np.int8)

    freqs_image = freqs_image[np.newaxis, :, :]
    month_channel = month_channel[np.newaxis, :, :]
    hour_channel = hour_channel[np.newaxis, :, :]
    stage_channel = stage_channel[np.newaxis, :, :]
    temperature_channel = temperature_channel[np.newaxis, :, :]

    freqs_image = np.concatenate((freqs_image, month_channel), axis=0)
    freqs_image = np.concatenate((freqs_image, hour_channel), axis=0)
    freqs_image = np.concatenate((freqs_image, stage_channel), axis=0)
    freqs_image = np.concatenate((freqs_image, temperature_channel), axis=0)
    return freqs_image


# (3) 전처리된 데이터에 대하여 pretrained deep ensemble 모델로 prediction 실행
def make_prediction(data_dir):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # WHERE THE LOG IS SAVED
    log_file = os.path.join('/workspace/demon', 'final_log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    test_set = DemonstrateDataset(data_dir)
    sample_test = DemonstrateTester()
    network = build_network('ensemble')
    network.load_state_dict(torch.load('/workspace/demon/final_weight'))
    sample_test.test(test_set, network)


class DemonstrateDataset(Dataset):

    def __init__(
        self, datadir
    ):
        super(Dataset, self).__init__()
        datafile = datadir

        with open(datafile, "r") as f:
            self.data = [line.strip().split('\t') for line in f.readlines()]

    def __getitem__(self, index):
        curr_data = np.array(self.data[index], dtype=np.float32)
        curr_X = curr_data[1:]
        curr_Y = np.int32(curr_data[0])

        curr_X_augmented = preprocess_data(curr_X)
        tensor_image = torch.tensor(curr_X_augmented, dtype=torch.float32)
        return tensor_image, curr_Y

    def __len__(self):
        return len(self.data)


class DemonstrateTester:

    def __init__(
        self, batch_size=128, device='cuda', n_jobs_dataloader=4
    ):
        self.batch_size = batch_size
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    def test(self, dataset, net):
        logger = logging.getLogger()

        # Get test data loader
        test_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_jobs_dataloader
        )

        # Set device for network
        net = net.to(self.device)

        criterion = nn.BCELoss()

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()

        net.eval()
        for epoch in range(1):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            for data in test_loader:
                inputs, targets, *_, = data
                targets = targets.long()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = net(inputs)

                loss = criterion(outputs, targets)
                epoch_loss += loss.item()

                # log epoch statistics
            epoch_test_time = time.time() - epoch_start_time
            logger.info(
                f'| Epoch: {epoch + 1:03} '
                f'| Test Time: {epoch_test_time:.3f}s '
                f'| Test Loss: {epoch_loss:.6f} |'
            )

        time_taken = time.time() - start_time
        logger.info('Testing Time: {:.3f}s'.format(time_taken))
        logger.info('Finished testing.')


if __name__ == '__main__':
    # e.g. /workspace/demon/testdir
    split_data('/workspace/lg_train_test')
    preprocess_data()
    make_prediction('/workspace/demon')
