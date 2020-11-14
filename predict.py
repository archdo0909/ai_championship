from model import build_network
from preprocessing import preprocess
from torch.utils.data import DataLoader
import torch
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, plot_roc_curve, auc
import time


# TODO: ensemble
def predict_ensemble():
    pass


# TODO: predict supervised models
def predict_supervised(model_path, data_path):
    net = build_network('resnet')
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # file open
    label_true = []
    label_pred = []
    f = open(data_path, 'r')
    header = f.readline() # ignore first line if header
    t_new = t_old = time.time()
    idx = 0
    while 1: 
        line = f.readline()
        if not line:
            break
        
        #label
        label_true.append(int(line[0]))
        # data
        line = line[1:].strip().split('\t')
        line[1] = int(line[1][1])
        data = np.array(line, dtype=np.float32)
        # preprocess each line
        freqs_image = preprocess(data)
        freqs_image = torch.Tensor(freqs_image).unsqueeze(0)

        # predict
        output = net(freqs_image)
        output = torch.argmax(output, 1)
        label_pred.append(int(output))
        # print(int(output))
        idx += 1
        if idx % 1000 == 0:
            t_new = time.time()
            print(idx, f'{(t_new - t_old):.2f}초 / 1000개, 초당 {(1000 / (t_new - t_old)):.2f}개')
            t_old = time.time()

    print('label_true : ', label_true)
    print('label_pred : ', label_pred)
    # ploting confusion matrix
    matrix = confusion_matrix(label_true, label_pred)
    print('confusion matrix : ', matrix)



    total=sum(sum(matrix))
    #####from confusion matrix calculate accuracy
    accuracy=(matrix[0,0]+matrix[1,1])/total
    print ('Accuracy : ', accuracy)

    sensitivity = matrix[0,0]/(matrix[0,0]+matrix[0,1])
    print('Sensitivity(recall) : ', sensitivity )

    specificity = matrix[1,1]/(matrix[1,0]+matrix[1,1])
    print('Specificity : ', specificity)

    precision = matrix[0,0]/(matrix[0,0]+matrix[1,0])
    print('Precision : ', precision)

    f1 = 2 * (precision * sensitivity) /  (precision+ sensitivity)
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



    y = np.array([1, 1, 2, 2])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, thresholds = roc_curve(label_true, label_pred, pos_label=2)
    roc_auc = auc(fpr, tpr)
    # fig, ax = plot_roc_curve(1, label_true, label_pred)
    plt.plot()
    plt.savefig('/workspace/roc_curve.png')


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

    # predict(model_path="/workspace/ai_championship/log/models/DeepSADModel.tar",
            # data_path="/workspace/ai_championship/data/sample_data.txt")
    # predict_supervised(model_path="/workspace/ai_championship/log/models/sample_train.pt",
    #         data_path="/workspace/ai_championship/data/sample_data.txt")
    predict_supervised(model_path="/workspace/ai_championship/log/models/sample_train.pt",
            data_path="/workspace/lg_train/202004_FLD165NBMA_vib_spectrum_modi_train.txt")