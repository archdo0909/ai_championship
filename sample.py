import numpy as np
import os

from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from glob import glob
import random
import time
import subprocess


class Sampler:
    """A sampler performs over and under samling on given dataset"""

    def __init__(self, src_path, src_path_abnormal, dst_path):
        self.file_to_load = src_path
        self.file_to_save = dst_path

        self.file_to_load_abnormal = src_path_abnormal
        
    def load_data(self, nb_data_to_load):
        # Load structured data given by LG Science Park 
        X = []
        y = []

        # Load normal data (정상 데이터)
        with open(self.file_to_load, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(X) == nb_data_to_load:
                    break
                curr_data = line.strip().split('\t')
                curr_data[2] = stage_value = int(curr_data[2][1])
                X.append(curr_data[1:])
                y.append(curr_data[0])

        # Load abnormal data (불량 데이터)
        with open(self.file_to_load_abnormal, 'r') as f:
            lines = f.readlines()
            for line in lines:
                curr_data = line.strip().split('\t')
                curr_data[2] = stage_value = int(curr_data[2][1])
                X.append(curr_data[1:])
                y.append(curr_data[0])

        # Possible type conversion required for sampling methods
        X_np = np.array(X).astype(np.float64)
        y_np = np.array(y).astype(np.int)
        return X_np, y_np

    def save_data(self, filename, X_resampled, y_resampled):
        with open(filename, 'a') as f:
            for data, label in zip(X_resampled, y_resampled):
                line = str(label) + '\t' + '\t'.join(map(str, data)) + '\n'
                f.write(line)

    def sample(self, nb_data_to_load, mode='combine'):
        X, y = self.load_data(nb_data_to_load)

        # Init sampler
        sampler = {
            'over': ADASYN(),
            'under': TomekLinks(),
            'combine': SMOTETomek(),
        }.get(mode)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Round datetime, stage and temperature
        X_resampled[:, 0] = X_resampled[:, 0].round()
        X_resampled[:, 1] = X_resampled[:, 1].round()
        X_resampled[:, 2] = X_resampled[:, 2].round(1)

        self.save_data(self.file_to_save, X_resampled, y_resampled)

    def random_sample(self, nb_data_to_load):

        X = []
        y = []

        # 한번에 돌면서 ng 데이터 따로 저장, 정상 데이터 샘플링 따로 저장

        # lg_train 폴더 안에 있는 파일을 하나씩 가져옴
        file_list = glob(self.file_to_load + '/*.txt')

        for filepath in file_list:
            print(filepath)

            # 파일의 총 데이터 개수 확인
            num_data = 0
            time_old = time.time()
            with open(filepath, mode='r') as f:
                for i, line in enumerate(f):
                    pass
            num_data = i+1
                
            # 데이터 개수를 range로 하는 random number 만들어서 list에 저장
            time_spend = time.time() - time_old
            print(f'time spend : {time_spend:.3f}')
            print('num_data:', num_data)
            
            random_index = random.sample(range(1,num_data),nb_data_to_load) # 1부터 num_data까지의 범위중에 nb_data_to_load개를 중복없이 뽑겠다.
            print(random_index)
            # list 안의 인덱스에 맞는 line 의 데이터 가져오기
            # 정상, 불량 데이터 X,y에 저장
            # Load normal data
            with open(filepath, mode='r') as f:
                for i, line in enumerate(f):
                    if i in random_index:
                        if line[0] == '0':
                            print("label : 0")
                            curr_data = line.strip().split('\t')
                            curr_data[2] = stage_value = int(curr_data[2][1])
                            X.append(curr_data[1:])
                            y.append(curr_data[0])
                        # print(line[0])
            
        # Load abnormal data (불량 데이터)
        with open(self.file_to_load_abnormal, 'r') as f:
            lines = f.readlines()
            for line in lines:
                curr_data = line.strip().split('\t')
                curr_data[2] = stage_value = int(curr_data[2][1])
                X.append(curr_data[1:])
                y.append(curr_data[0])

        # Possible type conversion required for sampling methods
        X_np = np.array(X).astype(np.float64)
        y_np = np.array(y).astype(np.int)


        sampler = SMOTETomek()

        X_resampled, y_resampled = sampler.fit_resample(X_np, y_np)

        # Round datetime, stage and temperature
        X_resampled[:, 0] = X_resampled[:, 0].round()
        X_resampled[:, 1] = X_resampled[:, 1].round()
        X_resampled[:, 2] = X_resampled[:, 2].round(1)

        self.save_data(self.file_to_save, X_resampled, y_resampled)



        # ng 데이터 샘플링
        # 80%만 추출하고
        # 나머지는 따로 저장
    

    def sample_all(self, nb_data_to_load):

        X = []
        y = []

        # 한번에 돌면서 ng 데이터 따로 저장, 정상 데이터 샘플링 따로 저장

        # lg_train 폴더 안에 있는 파일을 하나씩 가져옴
        file_list = glob(self.file_to_load + '/*.txt')

        for filepath in file_list:
            print(filepath)

            # list 안의 인덱스에 맞는 line 의 데이터 가져오기
            # 정상, 불량 데이터 X,y에 저장
            # Load normal data
            index = 0
            with open(filepath, mode='r') as f:
                for i, line in enumerate(f):
                    if line[0] == '0':
                        # print("label : 0")
                        curr_data = line.strip().split('\t')
                        curr_data[2] = stage_value = int(curr_data[2][1])
                        X.append(curr_data[1:])
                        y.append(curr_data[0])
                        index += 1
                        if index % 1000 == 0:
                            print(index)

        # Load random_700 data

        random_700 = '/workspace/peter/sampled/sampled_random700.txt'

        with open(random_700, mode='r')as f:
            if i in random_index:
                if line[0] == '0':
                    curr_data = line.strip().split('\t')
                    X.append(curr_data[1:])
                    y.append(curr_data[0])

        # Possible type conversion required for sampling methods
        X_np = np.array(X).astype(np.float64)
        y_np = np.array(y).astype(np.int)

        # Undersampling with Tomeklinks
        undersampler = TomekLinks()

        X_resampled, y_resampled = undersampler.fit_resample(X_np, y_np)


        # Round datetime, stage and temperature
        X_resampled[:, 0] = X_resampled[:, 0].round()
        X_resampled[:, 1] = X_resampled[:, 1].round()
        X_resampled[:, 2] = X_resampled[:, 2].round(1)

        self.save_data(self.file_to_save, X_resampled, y_resampled)



        # ng 데이터 샘플링
        # 80%만 추출하고
        # 나머지는 따로 저장




if __name__ == "__main__":
    # sampler = Sampler('', '', '')
    # sampler.sample(nb_data_to_load=100)
    # sampler = Sampler('/workspace/peter/lg_train', '/workspace/ng_train.txt', '/workspace/peter/sampled/sampled_random700.txt')
    # sampler.random_sample(nb_data_to_load = 700)

    sampler = Sampler('/workspace/peter/lg_train', '/workspace/ng_train.txt', '/workspace/peter/sampled/sampled_all.txt')
    sampler.sample_all(nb_data_to_load = 700)
