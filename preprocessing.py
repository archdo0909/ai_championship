import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Spectrogram(object):
     
    def __init__(self):
        pass
        
    def spec_array(self, arr):
        plt.rcParams["figure.figsize"] = (2.24, 2.24)
        plt.axis('off') # "invisable" axis in plot
        plt.xticks([]), plt.yticks([])
        plt.use_sticky_edges = True
        plt.margins(0)
        plt.specgram(list(arr), NFFT=10000, Fs=10, noverlap=5, detrend='mean', mode='psd')
        fig = plt.figure(1, tight_layout=True)
        fig.canvas.draw()
        fig.tight_layout(pad=0)

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape((3,) + fig.canvas.get_width_height()[::-1])

        return data