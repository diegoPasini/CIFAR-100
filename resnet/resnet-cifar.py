import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random


N = 20
BATCH_SIZE = 16
TRAINING_ITERATIONS = 100

mps_device = torch.device("mps")

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

meta_data = unpickle('/Users/diego/Scripts/CIFAR-100/cifar-100-python/meta')
test_data = unpickle('/Users/diego/Scripts/CIFAR-100/cifar-100-python/test')
train_data = unpickle('/Users/diego/Scripts/CIFAR-100/cifar-100-python/train')

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

print("SHAPE : ", raw_test_data.shape)

raw_test_data = torch.from_numpy(raw_test_data)
raw_train_data = torch.from_numpy(raw_train_data)


raw_test_data = raw_test_data.type(torch.FloatTensor).to(mps_device)
raw_train_data = raw_train_data.type(torch.FloatTensor).to(mps_device)


def get_batch(dataset_type = "train", batch_size = BATCH_SIZE):
    data = []
    labels = []
    if dataset_type == "train":
        random_indices = [random.randint(0, raw_train_data.shape[0]) for i in range(batch_size)]
        for i in random_indices:
            data.append(raw_train_data[i])
            labels.append(train_labels[i])
    elif dataset_type == "test":
        random_indices = [random.randint(0, raw_test_data.shape[0]) for i in range(batch_size)]
        for i in random_indices:
            data.append(raw_test_data[i])
            labels.append(test_labels[i])
    
    data = torch.stack(data) 
    labels = torch.tensor(labels, device=mps_device)
    return data, labels  



## The Hyperparams, as given in https://arxiv.org/pdf/1512.03385 for CIFAR-10
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual_core = nn.Sequential(
            # Dimensionality Reduction - Conv 1x1
            nn.Conv2d(in_channels, out_channels // 2, 1, padding=0),
            nn.BatchNorm2d(out_channels // 2),

            nn.LeakyReLU(),
            
            # Feature Extraction
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            
            nn.LeakyReLU(),
            
            # Dimensionality Expansion
            nn.Conv2d(out_channels // 2, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        
        if in_channels != out_channels:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = None
    
    def forward(self, x):
        residual = x
        x = self.residual_core(x)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        x = nn.LeakyReLU()(x + residual)
        return x

class ResNet(nn.Module):
    def __init__(self, n):
        super().__init__()

        # Implemented Incorrectly:
        self.conv_1 = Residual_Block(3, 16)
        self.list_1 = nn.ModuleList([Residual_Block(16, 16) for i in range(2 * n)])
        self.max_pool1 = nn.MaxPool2d(2, stride = 2)
        self.conv_2 = Residual_Block(16, 32)
        self.list_2 = nn.ModuleList([Residual_Block(32, 32) for i in range(2 * n - 1)])
        self.max_pool2 = nn.MaxPool2d(2, stride = 2)
        self.conv_3 = Residual_Block(32, 64)
        self.list_3 = nn.ModuleList([Residual_Block(64, 64) for i in range(2 * n)])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # There are 200 labels in CIFAR-100
        self.finLinLay = nn.Linear(64, 200)
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, x):
        x = self.conv_1(x)

        for block in self.list_1:
            x = block(x)
        x = self.max_pool1(x)
        x = self.conv_2(x)

        for block in self.list_2:
            x = block(x)

        x = self.max_pool2(x)
        x = self.conv_3(x)
        
        for block in self.list_3:
            x = block(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.finLinLay(x)
        x = self.softmax(x)
        return x

model = ResNet(N).to(mps_device)

total_params = sum(p.numel() for p in model.parameters())
print("Total Model Parameters :", total_params)

optimizer = torch.optim.Adam(model.parameters())
cross_entropy = nn.CrossEntropyLoss()


for i in tqdm(range(TRAINING_ITERATIONS)):
    optimizer.zero_grad()
    data, labels = get_batch("train")
    output = model(data)
    loss = cross_entropy(output, labels)
    loss.backward()
    print("Training Iteration: ", i, "Loss :", loss)
    optimizer.step()