import numpy as np


def preprocess(curr_data):
    curr_data = np.array(curr_data)

    freqs = curr_data[3:]
    freqs_image = freqs.reshape(100, -1)

    stage = curr_data[1]
    temperature = curr_data[2]

    stage_channel = np.full(freqs_image.shape, stage, dtype = np.int8)
    temperature_channel = np.full(freqs_image.shape, temperature, dtype = np.float)

    freqs_image = freqs_image[np.newaxis, :, :]
    stage_channel = stage_channel[np.newaxis, :, :]
    temperature_channel = temperature_channel[np.newaxis, :, :]
    # print('freq', freqs_image.shape)
    # print('stage', stage_channel.shape)
    # print('temp', temperature_channel.shape)

    freqs_image = np.concatenate((freqs_image, stage_channel), axis=0)
    freqs_image = np.concatenate((freqs_image, temperature_channel), axis=0)
    return freqs_image
