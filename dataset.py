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

        # 절대경로로 대체
        root = os.path.expanduser(root)

        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train
        self.folder_path = self.root / self.dataset_name
        self.label_path = self.folder_path / "label"
        self.stage_n_degc = stage_n_degc

        # catch label data
        fld = self.label_path.glob("**/*")
        file = list(fld)[0]

        """load label data

        input: "label_data.txt"
        column name:
            [label] [measure time] [stage] [temperature] [file_num]
            
        X contains [measure time] [stage] [temperature] [file_num]
        y contains [label]

        """
        data = []
        with open(file, "r") as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                data.append(line.strip().split("\t"))

        data = np.array(data)
        X = data[:, 1:]
        y = np.int32(data[:, 0])


        """
        split train, test and validation set
        train : 60%, test: 20%, validation: 20% 

        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.4,
                                                            random_state=random_state)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        test_size=0.5,
                                                        random_state=random_state)

        if self.train:
            self.data = X_train
            self.targets = torch.tensor(y_train)
        else:
            self.data = X_test
            self.targets = torch.tensor(y_test)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index)
        """
        if self.dataset_name == "lg_train":
            base_file_name = "_FLD165NBMA_vib_spectrum_modi_train_"

        sample, target = self.data[index], int(self.targets[index])
        measuretime = self.data[index][0]
        file_num = self.data[index][-1]

        if self.dataset_name == 'lg_train':
            # search Hz data
            target_fname = str(measuretime[:6]) + base_file_name + str(file_num) + ".txt"

        f = open(self.folder_path / target_fname, 'r')
        while 1:
            line = f.readline()
            if not line:
                break
            sample = line.strip().split('\t')[4:-1]
        f.close()

        target = int(self.targets[index])
        sample = list(map(float, sample))
        img_array = self.sp.spec_array(sample)

        if self.stage_n_degc:
            stage_layer, degc_layer = self.feature_layer(self.data[index][1:3])
            img_array = np.concatenate((img_array, stage_layer), axis=0)
            img_array = np.concatenate((img_array, degc_layer), axis=0)

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