import numpy as np


def preprocess(curr_data):
    freqs = curr_data[3:]
    freqs_image = freqs.reshape(100, -1)

    stage = curr_data[1]
    temperature = curr_data[2]

    stage_channel = np.full(freqs_image.shape, stage, dtype = np.int8)
    temperature_channel = np.full(freqs_image.shape, temperature, dtype = np.float)

    stage_channel = stage_channel[np.newaxis, :, :]
    temperature_channel = temperature_channel[np.newaxis, :, :]

    freqs_image = np.concatenate((freqs_image, stage_channel), axis=0)
    freqs_image = np.concatenate((freqs_image, temperature_channel), axis=0)
    return freqs_image
