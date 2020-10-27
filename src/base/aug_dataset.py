
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import torch

import numpy as np
from matplotlib import pyplot as plt


class TestDataset(Dataset):
    
    def __init__(self, root: str, dataset_name: str, train=True, random_state=None):
        super(Dataset, self).__init__()
        
        self.sp = Spectrogram()
        
        self.file_pth = root
        self.train = train
        self.random_state = None
        
        self.data = self.read_data()
        datasets = train_test_split(self.data)
        if train:
            self.train = datasets[0]
        else:
            # actually this is test dataset
            self.train = datasets[1]
    
    def __getitem__(self, index):
        
        # datasets[0][index][0]
        img = self.sp.spec_array(self.train[index][0])
        img = torch.tensor(img, dtype=torch.float32)
        target = self.train[index][1]
        target = torch.tensor(target)
        return img, target
    
    def __len__(self):
        return len(self.train)
    
    def read_data(self):
        datas = []
        #self.file = "./sample_data.txt"
        with open(self.file_pth, "r") as f:
            header = f.readline()
            while 1:
                line = f.readline()
                if not line:
                    break
                tmp = line.strip().split('\t')
                # freq = list(map(float, tmp[4:]))
                freq = list(map(float, tmp[1:]))
                # print(freq[-1])
                label = int(tmp[0])
                
                datas.append([freq,label])
                
        return datas


class Spectrogram(object):
    def __init__(self):
        pass
        
    def spec_array(self, arr):
        plt.rcParams["figure.figsize"] = (2.24,2.24)
        plt.axis('off') # "invisable" axis in plot
        plt.xticks([]), plt.yticks([])
        plt.use_sticky_edges = True
        plt.margins(0)
        plt.specgram(list(arr), NFFT=10000, Fs=10, noverlap=5, detrend='mean', mode='psd')
        fig = plt.figure(1, tight_layout=True)
        fig.canvas.draw()
        fig.tight_layout(pad=0)
    #     plt.close()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape((3,) + fig.canvas.get_width_height()[::-1])
    #     return np.array(fig.canvas.renderer._renderer)
        return data