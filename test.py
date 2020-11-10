from dataset import Testset
from model import build_network
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import SampleTrainer

trainer = SampleTrainer()



def inference_test(model_path, data_path, data_name):
    model = build_network('resnet')
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()


    # generate data set and data loader
    inference_set = Testset(root=data_path,
                          dataset_name=data_name,
                          train=True,
                          random_state=None,
                          stage_n_degc=True)
    inference_loader = DataLoader(inference_set, batch_size=trainer.batch_size, num_workers=trainer.n_jobs_dataloader)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in inference_loader:
            data = data.to("cuda")
            print(data.type)
            target = target.to("cuda")
            print(target.type)
            print(f"target : {target}")
            output = model(data)
            print(output.type)
            print(f"output : {output}")
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            print(f"pred : {pred}")
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(inference_loader.dataset),
                100.0 * correct / len(inference_loader.dataset),
            )
        )


if __name__ == "__main__":
    inference_test('/workspace/ai_championship/log/models/sample_train2.pt', '/workspace/ai_championship', 'data')