import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Resnet
from model import CRNN
from model import UNet
from model import LG_1DCNN

from preprocessing import preprocess_spectrogram


class EnsembleNetwork(nn.Module):
    def __init__(self):
        super(EnsembleNetwork, self).__init__()

        # Init dual class classifier
        self.resnet = Resnet()
        self.crnn = CRNN()
        self.unet = UNet()

        # Init one class classifier
        self.deep_sad_normal = LG_1DCNN()
        self.deep_sad_abnormal = LG_1DCNN()

        # Init models list
        self.models = [self.resnet, self.crnn, self.unet, self.deep_sad_normal, self.deep_sad_abnormal]

        # Load weights for non-anomaly detectors
        self.resnet.load_state_dict(torch.load('/workspace/jinsung/resnet_final-Copy1js.pt'))
        #self.crnn.load_state_dict(torch.load('/workspace/demon/crnn_random700_spectrogram.pt'))
        #self.unet.load_state_dict(torch.load('/workspace/demon/unet_random700_spectrogram.pt'))

        # Load DeepSAD Normal
        model_dict_normal = torch.load('/workspace/demon/deepSAD_1117_7k_10ep_64batch_normal_flip.tar')
        self.c_normal = model_dict_normal["c"]
        self.deep_sad_normal.load_state_dict(model_dict_normal["net_dict"])

        # Load DeepSAD Abnormal
        model_dict_abnormal = torch.load('/workspace/demon/deepSADModel_7k_10ep_64batch_abnormal.tar')
        self.c_abnormal = model_dict_abnormal["c"]
        self.deep_sad_abnormal.load_state_dict(model_dict_abnormal["net_dict"])

        # Load on CUDA and freeze parameter values
        for model in self.models:
            model.to('cuda')
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

    def forward(self, x):
        x_in_vec = torch.tensor(x[0, 3:-1], dtype=torch.float32, device='cuda')

        # Make prediction for DeepSAD models
        output_sad_normal = self.deep_sad_normal.forward(x_in_vec)
        distance_sad_normal = torch.sum((output_sad_normal - self.c_normal) ** 2, dim=1)
        score_sad_normal = round(torch.sqrt(distance_sad_normal).item())

        output_sad_abnormal = self.deep_sad_abnormal.forward(x_in_vec)
        distance_sad_abnormal = torch.sum((output_sad_abnormal - self.c_abnormal) ** 2, dim=1)
        score_sad_abnormal = round(torch.sqrt(distance_sad_abnormal).item())

        if score_sad_normal == score_sad_abnormal:
            return score_sad_normal
        
        # If not in consensus, try dual class classifiers
        x_np = x.cpu().detach().numpy().squeeze()
        x_in_img = preprocess_spectrogram(x_np)
        x_in_img = x_in_img[None, :, :, :]
        x_in_tensor = torch.tensor((x_in_img), dtype=torch.float32, device='cuda')
        result_resnet = self.resnet.forward(x_in_tensor)

        result_resnet = 1 if float(result_resnet) > 0.0001 else 0
        result_crnn = self.crnn.forward(x_in_tensor)
        result_unet = self.unet.forward(x_in_tensor)
        overall_result = 1 if float(result_resnet * 0.8 + result_crnn * 0.1 + result_unet * 0.1) else 0
        return overall_result


if __name__ == '__main__':
    # Create ensemble model
    model = EnsembleNetwork()

    # TODO: Load an actual input
    x = torch.randn(1, 5, 100, 100)

    # Make prediction
    output = model(x)
    print(output)
