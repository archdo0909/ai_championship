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


def make_prediction(data_dir=None, data_file=None, logfile=None):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # WHERE THE LOG IS SAVED
    log_file = os.path.join(data_dir, logfile)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    test_set = DemonstrateDataset(data_dir, data_file)
    demon_tester = DemonstrateTester()
    
    # Ensemble Pretrained Models
    ensemble_network = EnsembleNetwork()
    ensemble_network.to('cuda')
    demon_tester.test(test_set, ensemble_network)


class DemonstrateDataset(Dataset):

    def __init__(
        self, datadir, datafile
    ):
        super(Dataset, self).__init__()
        datafile = os.path.join(datadir, datafile)
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
        self, batch_size=1, device='cuda', n_jobs_dataloader=6
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
        logger.info('Start demonstration...')
        start_time = time.time()

        y_true = []
        y_pred = []
        for i, data in enumerate(test_loader):
            if i % 100 == 0:
                print('Current test index:', i, time.time() - start_time)
                
            x, y, *_ = data
            y = y.float()

            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = net(x)
            if not isinstance(y_hat, int):
                y_hat = y_hat.squeeze()
                y_hat_rounded = round(y_hat.data.cpu().item())
            else:
                y_hat_rounded = round(y_hat)

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
    datadir = '/workspace/demon'
    datafile = 'normal_100.txt'
    logfile = 'final_log_temp.txt'
    make_prediction(datadir, datafile, logfile)
