from torch.utils.data import DataLoader
from model import build_network
from dataset import LGDataset
from preprocessing import preprocess
from train import DeepSADTrainer

from utils.visualization.plot_images_grid import plot_images_grid

import time
import torch
import numpy as np


def predict(model_path, data_path):

    model_dict = torch.load(model_path, map_location="cpu")

    c = model_dict["c"]
    net = build_network("LG_LeNet")
    net.load_state_dict(model_dict["net_dict"])
    outlier_dist = model_dict["outlier_dist"]

    ae_net = build_network("LG_LeNet_Autoencoder")
    ae_net.load_state_dict(model_dict["ae_net_dict"])

    net.to("cuda")

    images = read_data(data_path)
    output = []
    for i in range(len(images)):
        outputs = net(torch.tensor(images[i], dtype=torch.float32).to("cuda"))
        dist = torch.sum((outputs - c) ** 2, dim=1)
        if dist > outlier_dist:
            label = 1
            print(print(f"label:{label}, abnormal"))
            print("dist : ", dist)
        else:
            label = 0
            print(print(f"label:{label}, normal"))
            print("dist : ", dist)

    return output


def test(model_path, dataset, batch_size, num_workers, eps, eta):

    model_dict = torch.load(model_path, map_location='cpu')

    c = model_dict['c']
    net = build_network('LG_LeNet')
    net.load_state_dict(model_dict['net_dict'])
    outlier_dist = model_dict['outlier_dist']

    ae_net = build_network('LG_LeNet_Autoencoder')
    ae_net.load_state_dict(model_dict['ae_net_dict'])
    print("Finish load net")
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    net.to('cuda')
    net.eval()

    epoch_loss = 0.0
    n_batches = 0
    start_time = time.time()
    idx_label_score = []
    outlier_dist = 0
    net.eval()
    print("start test")
    print(f"Size of test loader : {len(test_loader)}")
    with torch.no_grad():
        for data in test_loader:
            print(f"test rate : {n_batches} / {len(test_loader)}", end='\r', flush=True)
            inputs, labels, semi_targets, idx = data

            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            semi_targets = semi_targets.to('cuda')
            idx = idx.to('cuda')

            output = net(inputs)
            dist = torch.sum((output - c) ** 2, dim=1)
            losses = torch.where(semi_targets == 0, dist, eta * ((dist + eps)
                ** semi_targets.float()))
            loss = torch.mean(losses)
            scores = dist

            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                        labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

            epoch_loss += loss.item()
            n_batches += 1
            for label, score in zip(labels.cpu().data.numpy(), scores.cpu().data.numpy().tolist()):
                if not label:
                    if outlier_dist < score:
                        outlier_dist = score

    test_time = time.time() - start_time
    test_scores = idx_label_score

    _, labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)

    print("Test Loss : {:.6f}".format(epoch_loss / n_batches))
    print("Test time : {:.3f}s".format(test_time))

    return test_scores


def read_data(data_path):

    data = []
    f = open(data_path, "r")
    while 1:
        line = f.readline()
        if not line:
            break
        sample = line.strip().split("\t")[1:-1]
        data.append(sample)
    f.close()
    # data = list(map(float, data))

    img = []
    for i in range(len(data)):
        data[i] = list(map(float, data[i]))
        array = preprocess(data[i])
        img.append(array)

    return img


if __name__ == "__main__":

    predict(
        model_path="/workspace/eddie/ai_championship/log/models/deepSADModel.tar",
        data_path="/workspace/ai_championship/data/sample_data.txt",
    )

    test_set = LGDataset(root="/work/eddie_study/deep-sad-6k/data",
                         dataset_name="aug_6k",
                         train=False,
                         random_state=None,
                         stage_n_degc=False)

    test_scores = test(model_path="/work/eddie_study/deep-sad-6k/log/models/aug_6k_set.tar",
                       dataset=test_set,
                       batch_size=64,
                       num_workers=4,
                       eps=1e-6,
                       eta=0.01)

    indices, labels, scores = zip(test_scores)
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
    X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
    X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
    X_normal_high = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

    plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
    plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
    plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
    plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)
