import numpy as np

from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek


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


if __name__ == "__main__":
    sampler = Sampler('', '', '')
    sampler.sample(nb_data_to_load=100)
