from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import os
import torch

import numpy as np


class LGDataset(Dataset):

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        # 절대경로로 대체
        root = os.path.expanduser(root)

        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train
        self.folder_path = self.root / self.dataset_name
        self.label_path = self.folder_path / "label"
        
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
            self.data = X_train
            self.targets = torch.tensor(y_train)

    def __getitem__(self, index):
        """

        Args:
            index (int): Index
        
        Returns:
            tuple: (sample, target, index)
        """
        base_file_name = "_FLD165NBMA_vib_spectrum_modi_train_"

        sample, target = self.data[index], int(self.targets[index])
        measuretime = self.data[index][0]
        file_num = self.data[index][-1]
        
        # search Hz data
        target_fname = str(measuretime[:6]) + base_file_name + str(file_num) + ".txt"

        f = open(self.folder_path / target_fname, 'r')
        while 1:
            line = f.readline()
            if not line:
                break
            if measuretime in line: 
                sample = line.strip().split('\t')[5:]
        f.close()

        target = int(self.targets[index])
        img_array = self.spec_array(sample)
        sample = torch.tensor(img_array, dtype=torch.float64)

        return sample, target, index

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
        data = [np.genfromtxt(file, delimiter='\t', skip_header=1) for file in files]
        data = np.concatenate(data)

        # mapping stage information into int 
        X = data[:, 1:]
        y = data[:, 0]
        stage = data[:, 2]
        degc = data[:, 3]

        return X, y, stage, degc
    
    def spec_array(arr):
        plt.axis('off')
        plt.xticks([]), plt.yticks([])
        plt.use_sticky_edges = True
        plt.margins(0)
        plt.rcParams["figure.figsize"] = (2,2)
        plt.specgram(list(arr), NFFT=10000, Fs=10, noverlap=5, detrend='mean', mode='psd')
        fig = plt.figure(1, tight_layout=True)
        fig.canvas.draw()
        fig.tight_layout(pad=0)
     
        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
        return data

