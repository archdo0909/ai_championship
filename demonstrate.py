# -*- coding: utf-8 -*-

import os
import time
import torch
import logging

from glob import glob
import numpy as np
import torch.nn as nn

from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ensemble import EnsembleNetwork

from preprocessing import preprocess_spectrogram

# (1) 다운로드한 데이터(폴더)에서 정상과 불량 데이터 분리
def split_data(data_dir):
    # 불량 데이터 따로 빼고, 정상 데이터는 1000개씩 빼서 저장
    normal, abnormal = 0,0
    # num_extract_normal = 1000
    normal_data_fpath = os.path.join('/workspace/demon', 'normal.txt')
    abnormal_data_fpath = os.path.join('/workspace/demon', 'abnormal.txt')
    demonstration_data = os.path.join('/workspace/demon', 'demonstration_data.txt')

    file_list = glob(data_dir + '/*.txt')
    for filepath in file_list:
        print(filepath)
        with open(filepath, mode='r') as f:
            for i, line in enumerate(f):
                if line[0] == '0':
                    with open(demonstration_data, 'a') as f_all:
                        f_all.write(line)
                    with open(normal_data_fpath, 'a') as f_normal:
                        f_normal.write(line)
                        normal += 1
                elif line[0] == '1':
                    with open(demonstration_data, 'a') as f_all:
                        f_all.write(line)                
                    with open(abnormal_data_fpath, 'a') as f_abnormal:
                        f_abnormal.write(line)
                    abnormal += 1
    print(f'정상 데이터 : {normal} 개, 불량 데이터 {abnormal} 개')


# (2) 위에서 저장된 데이터에 대하여 데이터 전처리 실행 및 저장
def preprocess_data(curr_X=None):
    if curr_X is None:
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
def make_prediction(data_dir=None):
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
    demon_tester = DemonstrateTester()
    
    # Ensemble Pretrained Models
    ensemble_network = EnsembleNetwork()
    ensemble_network.to('cuda')
    demon_tester.test(test_set, ensemble_network)


class DemonstrateDataset(Dataset):

    def __init__(
        self, datadir
    ):
        super(Dataset, self).__init__()
        datafile = os.path.join(datadir, 'normal_100.txt')
        with open(datafile, "r") as f:
            self.data = [line.strip().split('\t') for line in f.readlines()]

    def __getitem__(self, index):
        data = self.data[index]
        data[2] = int(data[2][1])
        curr_data = np.array(data, dtype=np.float32)
        curr_X = curr_data[1:]
        curr_Y = np.int32(curr_data[0])

        #curr_X_augmented = preprocess_spectrogram(curr_X)
        #tensor_image = torch.tensor(curr_X_augmented, dtype=torch.float32)
        return curr_X, curr_Y

    def __len__(self):
        return len(self.data)


class DemonstrateTester:

    def __init__(
        self, batch_size=1, device='cuda', n_jobs_dataloader=4
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
        net.eval()

        # Testing
        logger.info('Start demonstration... god please...')
        start_time = time.time()

        y_true = []
        y_pred = []
        for i, data in enumerate(test_loader):
            if i % 50 == 0:
                print('Current test index:', i)
                
            x, y, *_ = data
            y = y.float()

            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = net(x).squeeze()
            y_hat_rounded = round(y_hat.data.cpu().item())

            y_true.append(int(y.data.cpu().item()))
            y_pred.append(y_hat_rounded)

        #print('y_true: {}, y_pred: {}'.format(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        print('Confusion Matrix:')
        print(cm)

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1_score = 2*precision*recall / (precision+recall)
        balanced_accuracy = (recall + specificity) / 2
        print('Precision: {}, Recall: {}, Specificity: {}, F1-Score: {}, Balanced Accuracy: {}'.format(
            precision, recall, specificity, f1_score, balanced_accuracy
        ))

        time_taken = time.time() - start_time
        logger.info('Time taken for making prediction: {:.3f}s'.format(time_taken))
        logger.info('Finished demonstration.')


if __name__ == '__main__':
    # e.g. /workspace/demon/testdir
    #split_data('/workspace/test1')
    preprocess_data()
    make_prediction('/workspace/demon')
