

import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_network


class SampleTester:
    
    def __init__(self, lr: float = 0.001, n_epochs: int = 150, batch_size: int = 128,
                 device: str = 'cuda', n_jobs_dataloader: int = 0):
        
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
    
    def test(self, dataset, net):
        logger = logging.getLogger()
        
        # Get test data loader
        test_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        
        # Set device for network
        net = net.to(self.device)

        criterion = nn.CrossEntropyLoss()

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()

        net.eval()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in test_loader:
                inputs, targets, _, _ = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = net(inputs)

                loss = criterion(outputs, targets)
                epoch_loss += loss.item()
                n_batches += 1 

            # log epoch statistics
            epoch_test_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Test Time: {epoch_test_time:.3f}s '
                        f'| Test Loss: {epoch_loss / n_batches:.6f} |')

        self.test_time = time.time() - start_time
        logger.info('Testing Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

        torch.save(net.state_dict(), "/workspace/ai_championship/log/models/sample_test.pt")
        return net


if __name__ == "__main__":
    inference_test('/workspace/ai_championship/log/models/sample_train2.pt', '/workspace/ai_championship', 'data')