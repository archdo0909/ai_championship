from torch.utils.data import DataLoader
from model import build_network
from dataset import LGDataset
from preprocessing import preprocess
from train import DeepSADTrainer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, roc_curve, plot_roc_curve, auc

import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def predict(model_path, data_path):

    model_dict = torch.load(model_path, map_location="cpu")

    c = model_dict["c"]
    net = build_network("LG_LeNet")
    net.load_state_dict(model_dict["net_dict"])
    outlier_dist = model_dict["outlier_dist"]

    ae_net = build_network("LG_LeNet_Autoencoder")
    ae_net.load_state_dict(model_dict["ae_net_dict"])

    net.to("cuda")

    images, label_trues = read_data(data_path)
    label_preds = []
    for i in range(len(images)):
        outputs = net(torch.tensor(images[i], dtype=torch.float32).to("cuda"))
        dist = torch.sum((outputs - c) ** 2, dim=1)
        if dist > outlier_dist:
            label_preds.append(1)
        else:
            label_preds.append(0)

    evaluate(label_trues, label_preds)


def evaluate(label_true, label_pred):

    matrix = confusion_matrix(label_true, label_pred)

    total = sum(sum(matrix))
    # from confusion matrix calculate accuracy
    accuracy = (matrix[0, 0] + matrix[1, 1]) / total
    print('Accuracy : ', accuracy)

    sensitivity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    print('Sensitivity(recall) : ', sensitivity)

    specificity = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])
    print('Specificity : ', specificity)

    precision = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    print('Precision : ', precision)

    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    print('F1 Score : ', f1)

    print("\n\nSensitivity or recall: TP / (TP + FN)     // 맞는 케이스에 대해 얼마나 많이 맞다고 실제로 예측했나?")
    print("Specificity: TN / (FP + TN)     // 틀린 케이스에 대해 얼마나 틀리다고 실제로 예측했나?")
    print("Precision: TP / (TP + FP)     // 맞다고 예측 한 것 중에 실제로 정답이 얼마나 되나?")

    fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                    show_absolute=True,
                                    show_normed=True,
                                    colorbar=True)
    plt.plot()
    plt.savefig('/workspace/confusion.png')


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
            losses = torch.where(semi_targets == 0, dist, eta * ((dist + eps)** semi_targets.float()))
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

    label_true = []
    data = []
    f = open(data_path, 'r')

    img = []

    while 1:
        line = f.readline()
        if not line:
            break

        # label
        label_true.append(int(line[0]))
        # data
        line = line[1:].strip().split('\t')
        line[1] = int(line[1][1])
        data = np.array(line, dtype=np.float32)

        freqs_image = preprocess(data)
        img.append(freqs_image)
        img = torch.Tensor(img).unsqueeze(0)

    return img, label_true


if __name__ == "__main__":

    predict(
        model_path="/workspace/eddie/deep-sad-6k/log/models/aug_6k_set.tar",
        data_path="/workspace/ng.txt",
    )

    # test_set = LGDataset(root="/workspace/eddie/deep-sad-6k/data",
    #                      dataset_name="sampled",
    #                      train=False,
    #                      random_state=None,
    #                      stage_n_degc=False)

    # test_scores = test(model_path="/workspace/eddie/deep-sad-6k/log/models/aug_6k_set.tar",
    #                    dataset=test_set,
    #                    batch_size=16,
    #                    num_workers=4,
    #                    eps=1e-6,
    #                    eta=0.01)