from pathlib import Path
from torch.utils.data import Dataset

import os
import torch

import numpy as np
import pandas as pd

class LGDataset(Dataset):

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        # 절대경로로 대체
        root = os.path.expanduser(root)

        self.root = Path(root)
        self.dataset_name = datset_name
        self.train = train
        self.folder_path = self.root / self.dataset_name

        # Read all data
        X, y, stage, degc = read_data()

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

    def __getitem__(self, index):
        """

        Args:
            index (int): Index
        
        Returns:
            tuple: (sample, target, index)
        """


        return sample, target

    def __len__(self):
        return len(self.data)

    def read_data(self):
        """
        Returns:
            X: Hz data
            y: label data
            stage: working stage
            degc: temperature in environment
        """
        fld = self.folder_path.glob("**/*")
        files = [x for x in fld if os.path.isfile(x)]
        # read files with pandas
        # df = pd.concat([pd.read_csv(file, sep='\t') for file in files if os.path.isfile(file)],
        #                ignore_index=True)

        # y = df['label']
        # stage = df['stage']
        # degc = df['degc']
        # read files with numpy
        data = [np.genfromtxt(file, delimiter='\t', skip_header==1) for file in files]
        data = np.concatenate(data)
        length = len(data)
        
        X = data[:][4:]
        y = data[:][0]
        stage = data[:][2]
        degc = data[:][3]

        return X, y, stage, degc
        

