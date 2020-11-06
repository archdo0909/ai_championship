from torch.utils.data import DataLoader
from dataset import LGDataset
from train import DeepSADTrainer, AETrainer
from model import build_network

import torch
import logging


def main(xp_path, network, optimizer_name, c, eta, lr, n_epochs, batch_size, lr_milestones, weight_decay,
         ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         device, n_jobs_dataloader, stage_n_degc=None):
    """
        xp_path : 결과물 출력할 폴더의 절대 경로
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    train_set = LGDataset(root='/workspace/ai_championship/data',
                          dataset_name='lg_train',
                          train=True,
                          random_state=None,
                          stage_n_degc=False)

    ae_train = AETrainer(ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size,
                         ae_weight_decay, device, n_jobs_dataloader)

    deep_sad_train = DeepSADTrainer(c, eta, optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay,
                                    device, n_jobs_dataloader)
    
    # Register your network in model.py
    ae_net = build_network('LG_LeNet_Autoencoder')
    net = build_network('LG_LeNet')
    ae_net = ae_train.train(train_set, ae_net)

    net_dict = net.state_dict()
    ae_net_dict = ae_net.state_dict()

    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
    # Overwrite values in the existing state_dict
    net_dict.update(ae_net_dict)
    # Load the new state_dict
    net.load_state_dict(net_dict)

    net = deep_sad_train.train(train_set, net)


if __name__ == "__main__":

    main(xp_path='/workspace/ai_championship/log',
         network='Peter_CNN',
         optimizer_name='Adam',
         c=0.01,
         eta=0.01,
         lr=0.001,
         n_epochs=1,
         batch_size=10,
         lr_milestones=(5,),
         weight_decay=0.5e-3,
         ae_optimizer_name='Adam',
         ae_lr=0.001,
         ae_n_epochs=1,
         ae_lr_milestone=(5,),
         ae_batch_size=10,
         ae_weight_decay=0.5e-3,
         device='cuda',
         n_jobs_dataloader=4,
         stage_n_degc=True)