from torch.utils.data import DataLoader, Subset
from dataset import LGDataset
from train import DeepSADTrainer, AETrainer
from model import build_network
from preprocessing import create_semisupervised_setting

import torch
import logging
import warnings

warnings.filterwarnings("ignore")


def main(root, dataset_name, output_model_name, xp_path, network, optimizer_name, c, eta, lr, n_epochs, batch_size, lr_milestones, weight_decay,
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
    export_model = xp_path + '/models/DeepSADModel.tar'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # train_set = LGDataset(root='/workspace/eddie/ai_championship/data',
    #                       dataset_name='lg_train',
    #                       train=True,
    #                       random_state=None,
    #                       stage_n_degc=False)
    train_set = LGDataset(root=root,
                          dataset_name=dataset_name,
                          train=True,
                          random_state=None,
                          stage_n_degc=False)
    
    idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(),
                                                         normal_classes=(0,),
                                                         outlier_classes=(1,),
                                                         known_outlier_classes=(),
                                                         ratio_known_normal=0,
                                                         ratio_known_outlier=0, 
                                                         ratio_pollution=0)

    train_set.semi_targets[idx] = torch.tensor(semi_targets, dtype=torch.int32)
    train_set = Subset(train_set, idx)

    # test_set = LGDataset(root='/workspace/eddie/ai_championship/data',
    #                      dataset_name='lg_train',
    #                      train=False,
    #                      random_state=None,
    #                      stage_n_degc=False)
    test_set = LGDataset(root=root,
                         dataset_name=dataset_name,
                         train=False,
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

    net, c = deep_sad_train.train(train_set, net)

    outlier_dist = deep_sad_train.test(test_set, net)

    # save model
    net_dict = net.state_dict()
    ae_net_dict = ae_net.state_dict()

    torch.save({
        'c': c,
        'net_dict': net_dict,
        'ae_net_dict': ae_net_dict,
        'outlier_dist': outlier_dist
    }, export_model)
    

if __name__ == "__main__":

    main(root='/workspace/eddie/ai_championship/data',
         dataset_name='aug_6k',
         output_model_name='deepSADModel_6k.tar',
         xp_path='/workspace/eddie/ai_championship/log',
         network='LG',
         optimizer_name='Adam',
         c=0.01,
         eta=0.01,
         lr=0.01,
         n_epochs=30,
         batch_size=5,
         lr_milestones=(2, 5,),
         weight_decay=0.5e-3,
         ae_optimizer_name='Adam',
         ae_lr=0.1,
         ae_n_epochs=30,
         ae_lr_milestone=(2, 5,),
         ae_batch_size=5,
         ae_weight_decay=0.5e-2,
         device='cuda',
         n_jobs_dataloader=4,
         stage_n_degc=True)
