import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re




def spec_array(arr):
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

def feature_layer(features):
    # Input : feature list ( index 0 : stage, index 1 : degc)
    # Output : feature layer - 224*224 사이즈의 stage, degc 레이어 각각 1개씩

    stage = int(re.findall("\d+", features[0])[0])
    degc = float(features[1])

    stage_layer = np.full((224,224), stage, dtype = np.int8)
    degc_layer = np.full((224,224), degc, dtype = np.float)

    stage_layer = stage_layer[np.newaxis,:,:]
    degc_layer = degc_layer[np.newaxis,:,:]

    return stage_layer, degc_layer


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

