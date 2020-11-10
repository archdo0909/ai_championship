# -*- coding: utf-8 -*-
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from preprocessing import Spectrogram

import os
import re
import torch

import numpy as np


class LGDataset(Dataset):

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None, stage_n_degc: str=None):
        super(Dataset, self).__init__()

        self.sp = Spectrogram()

        self.classes = [0, 1]
        
        self.root = root
        self.dataset_name = dataset_name
        self.train = train
        self.random_state = random_state
        self.stage_n_degc = stage_n_degc

        self.dataset_path = self.root + '/' + self.dataset_name

        tmp_file_list = os.listdir(self.dataset_path)
        tmp_file_list.sort()
        file_list = []
        for files in tmp_file_list: 
            if files.endswith('.txt'):
                file_list.append(files)

        data = np.array(file_list)

        target_data = []
        label_path = self.dataset_path + "/" + "label" + "/" + "label_data.txt"
        print("reading data")
        with open(label_path, 'r') as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                sample = line.strip().split('\t')
                target_data.append(sample)
            f.close()
        
        target_data = np.array(target_data)
        X = data
        y = np.int32(target_data[:,0])
        X_train, X_test, y_train, y_test = train_test_split(data, y,
                                                            test_size=0.4,
                                                            random_state=random_state)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        test_size=0.5,
                                                        random_state=random_state)
        if self.train:
            self.data = X_train
            self.targets = torch.tensor(y_train)
            #self.targets = torch.tensor(np.ones(self.data.shape, dtype=np.float64))
        else:
            self.data = X_test
            self.targets = torch.tensor(y_test)
            #self.targets = torch.tensor(np.ones(self.data.shape, dtype=np.float64))
        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index)
        """
        self.data_name = self.dataset_path + "/" + self.data[index]

        f = open(self.data_name, 'r')
        while 1:
            line = f.readline()
            if not line:
                break
            sample = line.strip().split('\t')
        f.close()

        target = int(sample[0])
        freq = list(map(float, sample[4:]))
        img_array = self.sp.spec_array(freq)

        sample = torch.tensor(img_array, dtype=torch.float32)
        semi_targets = int(self.semi_targets[index])

        return sample, target, semi_targets, index

    def __len__(self):
        return len(self.data)

    def feature_layer(self, features):
        
        stage = int(re.findall("\d+", features[0])[0])
        degc = float(features[1])

        stage_layer = np.full((224,224), stage, dtype = np.int8)
        degc_layer = np.full((224,224), degc, dtype = np.float)

        stage_layer = stage_layer[np.newaxis,:,:]
        degc_layer = degc_layer[np.newaxis,:,:]

        return stage_layer, degc_layer
