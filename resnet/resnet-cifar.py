import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import random
import os
import wandb


# N = 60
BATCH_SIZE = 256
# NUM_EPOCS = 150
TRAINING_ITERATIONS = 50
WANDB_PROJECT_NAME = "cifar-100"

resnet_sizes = [50]
learning_rates = [0.1]
num_epochs = [200]

cuda_device = torch.device("cuda")
checkpoint_dir = '/root/CIFAR-100/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir = checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')

meta_data = unpickle('/root/CIFAR-100/cifar-100-python/meta')
test_data = unpickle('/root/CIFAR-100/cifar-100-python/test')
train_data = unpickle('/root/CIFAR-100/cifar-100-python/train')

fine_label_names = meta_data[b'fine_label_names']
print("Number of Labels: ", len(fine_label_names))
coarse_label_names = meta_data[b'coarse_label_names']
print("Number of Corase Labels: ", len(coarse_label_names))

raw_test_data = test_data[b'data']
test_labels = test_data[b'fine_labels']
print("Test Raw Image Data Shape: ", raw_test_data.shape)
print("Test Labels Shape: ", len(test_labels))

raw_train_data = train_data[b'data']
train_labels = train_data[b'fine_labels']
print("Raw Training Data Shape: ", raw_train_data.shape)
print("Train Labels Shape", len(train_labels))

label_from_index = lambda index : fine_label_names[index]
index_from_label = lambda label : fine_label_names.index(label)

raw_test_data = raw_test_data.reshape(raw_test_data.shape[0], 3, 32, 32)
raw_test_data = raw_test_data.astype('uint8')

raw_train_data = raw_train_data.reshape(raw_train_data.shape[0], 3, 32, 32)
raw_train_data = raw_train_data.astype('uint8')

raw_test_data = torch.from_numpy(raw_test_data)
raw_train_data = torch.from_numpy(raw_train_data)

raw_test_data = raw_test_data.type(torch.FloatTensor).to(cuda_device)
raw_train_data = raw_train_data.type(torch.FloatTensor).to(cuda_device)

class DatasetCIFAR100(Dataset):
    def __init__(self, raw_data, labels):
        self.raw_data = raw_data
        self.labels = labels
        self.mean = np.mean(raw_data)
        self.std = np.std(raw_data)
    
    def __len__(self):
        return len(train_labels)

    def __getitem__(self, idx):
        data = self.raw_data[idx]
        transforms = v2.Compose([
            v2.RandomResizedCrop(size=32, padding = 2),
            v2.RandomHorizontalFlip(p = 0.5),
            v2.RandomRotation(15),
            v2.ToTensor(),
            v2.Normalize(self.mean, self.std),
        ])  

        data = transforms(data)
        return data
    
train_dataset = DatasetCIFAR100(raw_train_data, train_labels)
test_dataset = DatasetCIFAR100(raw_test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

print("SHAPE : ", raw_test_data.shape)

# def get_batch(dataset_type = "train", batch_size = BATCH_SIZE):
#     data = []
#     labels = []
#     if dataset_type == "train":
#         random_indices = [random.randint(0, raw_train_data.shape[0] - 1) for i in range(batch_size)]
#         for i in random_indices:
#             data.append(raw_train_data[i])
#             labels.append(train_labels[i])
#     elif dataset_type == "test":
#         random_indices = [random.randint(0, raw_test_data.shape[0] - 1) for i in range(batch_size)]
#         for i in random_indices:
#             data.append(raw_test_data[i])
#             labels.append(test_labels[i])
    
#     data = torch.stack(data) 
#     labels = torch.tensor(labels, device=cuda_device)
#     return data, labels  



def evaluate(model, loss_func):
    loss_avg = 0
    correct_predictions = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        num_batches = int(raw_test_data.shape[0] // BATCH_SIZE * 0.5)
        print(num_batches)
        for i in range(num_batches):
            batch, labels = next(iter(test_loader))
            output = model(batch)
            loss_avg += loss_func(output, labels).item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    loss_avg /= num_batches
    accuracy_avg = (correct_predictions / total_samples) * 100
    return loss_avg, accuracy_avg


## The Hyperparams, as given in https://arxiv.org/pdf/1512.03385 for CIFAR-10
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.residual_core = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.residual_core(x)
        out += self.shortcut(x)
        return nn.ReLU(inplace=True)(out)


class ResNet(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage_1 = nn.ModuleList([Residual_Block(64, 64) for _ in range(2)])
        self.stage_2 = nn.ModuleList([Residual_Block(64, 128, stride=2)] + [Residual_Block(128, 128) for _ in range(2)])
        self.stage_3 = nn.ModuleList([Residual_Block(128, 256, stride=2)] + [Residual_Block(256, 256) for _ in range(2)])
        self.stage_4 = nn.ModuleList([Residual_Block(256, 512, stride=2)] + [Residual_Block(512, 512) for _ in range(2)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 100)
    
    def forward(self, x):
        x = self.conv_1(x)
        for block in self.stage_1:
            x = block(x)
        for block in self.stage_2:
            x = block(x)
        for block in self.stage_3:
            x = block(x)
        for block in self.stage_4:
            x = block(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

for resnet_size in resnet_sizes:
    for lr in learning_rates:
        for epoch in num_epochs:
            model = ResNet(resnet_size).to(cuda_device)
            run_name = f"ResNet{resnet_size}_LR{lr}_BS{BATCH_SIZE}_EP{epoch}"
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total Model Parameters (ResNet-{resnet_size}): {total_params}")

            wandb.init(project=WANDB_PROJECT_NAME, name = run_name)

            wandb.config.update({
                "resnet_size": resnet_size,
                "learning_rate": lr,
                "batch_size": BATCH_SIZE,
                "num_epochs": epoch,
                "scheduler_patience": 5,
                "scheduler_factor": 0.5,
                "model_parameters": total_params
            })

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            cross_entropy = nn.CrossEntropyLoss()

            for j in range(epoch):
                print("Epoch: ", j)
                train_acc = 0
                model.train()
                for i in tqdm(range(TRAINING_ITERATIONS)):
                    optimizer.zero_grad()
                    data, labels = next(iter(train_loader))
                    output = model(data)
                    loss = cross_entropy(output, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step(loss)
                    _, predicted = torch.max(output, 1)
                    train_acc += (predicted == labels).sum().item()
                    del data, labels, output
                    torch.cuda.empty_cache()

                final_train_acc = train_acc / (TRAINING_ITERATIONS * BATCH_SIZE) * 100
                print("Training Loss: ", loss, "Training Accuracy: ", final_train_acc)
                loss_test, accuracy = evaluate(model, cross_entropy)
                print("Test Loss: ", loss_test, "Test Accuracy: ", accuracy)
                wandb.log({
                    "epoch": j,
                    "training_loss": loss,
                    "training_accuracy": final_train_acc,
                    "test_loss": loss_test,
                    "test_accuracy": accuracy
                })
                save_checkpoint(model, optimizer, j, loss, accuracy)
                torch.cuda.empty_cache()

            wandb.finish()