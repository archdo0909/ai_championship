import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Spectrogram(object):
    def __init__(self):
        pass


    def preprocessing(self, filename):
        data = pd.read_csv(filename, sep='\t')
        spectrum = data.loc[:, '0 Hz':'10000 Hz']

        return spectrum




    def graph(self, data):
        length = len(data)
        plt.rcParams["figure.figsize"] = (14, 2*length//2)
        specs = np.array(data)
        for i in range(len(specs)):
            plt.subplot(length//2, 2, i + 1)
            plt.plot(specs[i])
        plt.show()

    def spectrogram(self, data):
        length = len(data)
        plt.rcParams["figure.figsize"] = (14, 2 * length // 2)
        specs = np.array(data)
        for i in range(len(specs)):
            plt.subplot(length//5, 5, i + 1)
            plt.specgram(list(specs[i]), NFFT=10000, Fs=10, noverlap=5)
            plt.xlabel('time')
            plt.ylabel('frequency')
        plt.show()



def main():
    sp = Spectrogram()
    d = sp.preprocessing("./sample_data.txt")
    sp.graph(d)
    # sp.spectrogram(d)



if __name__ == "__main__":
    main()
