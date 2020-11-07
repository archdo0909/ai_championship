# -*- coding: utf-8 -*-
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from preprocessing import spec_array, feature_layer

import os
import re
import torch

import numpy as np


class LGDataset(Dataset):

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None, stage_n_degc: str=None):
        super(Dataset, self).__init__()

        self.root = root
        self.dataset_name = dataset_name
        self.train = train
        self.random_state = random_state
        self.stage_n_degc = stage_n_degc

        self.classes = [0, 1]

        self.dataset_path = self.root + self.dataset_name
        file_list = []
        
        for files in os.listdir(self.dataset_path):
            if files.endswith('.txt'):
                file_list.append(files)

        data = np.array(file_list)

        X_train, X_test, y_train, y_test = train_test_split(data, data,
                                                            test_size=0.4,
                                                            random_state=random_state)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        test_size=0.5,
                                                        random_state=random_state)

        if self.train:
            self.data = X_train
            #self.targets = y_train
        else:
            self.data = X_test
            #self.targets = y_test

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index)
        """
        self.data_name = self.data[index]
        f = open(self.dataset_path / self.data_name, 'r')
        while 1:
            line = f.readline()
            if not line:
                break
            sample = line.strip().split('\t')
        f.close()

        target = int(sample[0])
        freq = list(map(float, sample[4:]))
        img_array = spec_array(freq)

        if self.stage_n_degc:
            stage_layer, degc_layer = feature_layer(self.data[index][1:3])
            img_array = np.concatenate((img_array, stage_layer), axis=0)
            img_array = np.concatenate((img_array, degc_layer), axis=0)

        sample = torch.tensor(img_array, dtype=torch.float32)
        semi_targets = int(self.semi_targets[index])

        return sample, target, semi_targets, index

    def __len__(self):
        return len(self.data)
