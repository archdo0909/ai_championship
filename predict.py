from model import build_network
from preprocessing import Spectrogram

import torch
import numpy as np


def predict(model_path, data_path):

    model_dict = torch.load(model_path, map_location='cpu')

    c = model_dict['c']
    net = build_network('LG_LeNet')
    net.load_state_dict(model_dict['net_dict'])
    outlier_dist = model_dict['outlier_dist']
    
    ae_net = build_network('LG_LeNet_Autoencoder')
    ae_net.load_state_dict(model_dict['ae_net_dict'])

    net.to('cuda')

    images = read_data(data_path)
    output = []
    for i in range(len(images)):
        outputs = net(torch.tensor(images[i], dtype=torch.float32).to('cuda'))
        dist = torch.sum((outputs - c) ** 2, dim=1)
        if dist > outlier_dist:
            label = 1
            print(print(f"label:{label}, abnormal"))
        else:
            label = 0
            print(print(f"label:{label}, normal"))

    # print(f"Show output : {output}")

    return output


def read_data(data_path):

    sp = Spectrogram()

    data = []
    f = open(data_path, 'r')
    while 1: 
        line = f.readline()
        if not line:
            break
        sample = line.strip().split('\t')[4:-1]
        data.append(sample)
    f.close()
    # data = list(map(float, data))

    img = []
    for i in range(len(data)):
        data[i] = list(map(float, data[i]))
        array = sp.spec_array(data[i])
        img.append(array)

    return img


if __name__ == "__main__":

    predict(model_path="/workspace/ai_championship/log/models/DeepSADModel.tar",
            data_path="/workspace/ai_championship/data/sample_data.txt")