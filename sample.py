import numpy as np

from imblearn.over_sampling import SMOTE, SMOTENC


class Sampler:
    """A sampler performs over and under samling on given dataset"""

    def __init__(self, src_path, dst_path):
        self.file_to_load = src_path
        self.file_to_save = dst_path
        
    def load_data(self, nb_data_to_load):
        # Load structured data given by LG Science Park 
        X = []
        y = []
        with open(self.file_to_load, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(X) == nb_data_to_load:
                    break
                curr_data = line.strip().split('\t')
                curr_data[2] = stage_value = int(curr_data[2][1])
                X.append(curr_data[1:])
                y.append(curr_data[0])
        return X, y

    def oversample(self, nb_data_to_load):
        X, y = self.load_data(nb_data_to_load)

        # Type conversion is required for ADASYN
        X_np = np.array(X).astype(np.float64)
        y_np = np.array(y).astype(np.int)
        
        # Oversamling data with SMOTENC
        oversampler = SMOTENC(categorical_features=[2] , random_state=0)
        X_resampled, y_resampled = oversampler.fit_sample(X_np, y_np)
        
        # Save oversampled data to a text file
        with open(self.file_to_save, 'a') as f:
            for data, label in zip(X_resampled, y_resampled):
                line = str(label) + '\t' + '\t'.join(map(str, data)) + '\n'
                f.write(line)

    def undersample(self, nb_data_to_load):
        raise NotImplementedError


if __name__ == "__main__":
    sampler = Sampler('', '')
    sampler.oversmaple(100)
