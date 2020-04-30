#!/usr/bin/env python
# coding: utf-8

# libraries setup
import torch
import torchvision

import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from glob import glob
from PIL import Image
import dload

# model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=200,kernel_size=(7,7),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=200,out_channels=250,kernel_size=(4,4),padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=250, out_channels=350, kernel_size=(4,4), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(350 * 6 * 6, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 43),
        )

    def forward(self,x):
        # Conv layers
        x = self.conv_layer1(x)
        # LCN layer
        x = local_contrast_normalization(x)
        # Conv layers
        x = self.conv_layer2(x)
        # LCN layer
        x = local_contrast_normalization(x)
        # Conv layers
        x = self.conv_layer3(x)
        # LCN layer
        x = local_contrast_normalization(x)
        # flatten
        x = x.view(-1,350 * 6 * 6)
        # FC layer
        x = self.fc_layer(x)

def local_contrast_normalization(in_tensor):
    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
    channels = in_tensor.shape[1]
    x = np.zeros((1,channels,9,9), dtype='float64')
    for channel in range(channels):
        for i in range(9):
            for j in range(9):
                x[0,channel,i,j] = gauss(i - 4, j - 4)
    gaussian_filter = torch.Tensor(x/np.sum(x))
    filtered = F.conv2d(in_tensor, gaussian_filter, bias=None, padding=8)
    mid = int(np.floor(gaussian_filter.shape[2]/2.))
    centered_image = in_tensor - filtered[:, :, mid:-mid, mid:-mid]
    sum_sqr_image = F.conv2d(centered_image.pow(2),gaussian_filter,bias=None,padding=8)
    s_deviation = sum_sqr_image[:,:,mid:-mid,mid:-mid].sqrt()
    per_img_mean = s_deviation.mean()
    divisor = torch.Tensor(np.maximum(np.maximum(per_img_mean.numpy(),s_deviation.numpy()),1e-4))
    out_tensor = centered_image/divisor
    return out_tensor

# dataset
class GTSRB(data.Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.data = []
        self.transform = transform
        class_folders_paths = sorted(glob(os.path.join("{}/Final_Training/Images".format(self.path), "*")))
        for folders_paths in class_folders_paths:
            folder_img_paths = sorted(glob(os.path.join(folders_paths, "*.ppm")))
            self.data.extend([(_,int(folders_paths.split("/")[-1])) for _ in folder_img_paths])

    def __getitem__(self, i):
        img_path, label = self.data[i]
        im = Image.open(img_path)
        if self.transform: im = self.transform(im)
        return im, label

    def __len__(self): return len(self.data)

    def get_orig_img_path(self,i): return self.data[i]

def get_transform():
    return T.Compose([
        T.Resize((48,48)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

# training and evaluation pipeline
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for i, (data_batch, batch_labels) in enumerate(train_loader):
        preds = model(data_batch)
        loss = criterion(preds, batch_labels.to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(model, data_loader, device):
    y_true, y_pred = [], []

    model.eval()
    for data_batch, batch_labels in data_loader:
        preds = model(data_batch.to_device()).cpu()
        y_pred.extend(list(preds))
        y_true.extend(list(batch_labels))

    accuracy = accuracy_score(y_true, y_pred, normalize=True)
    
    return accuracy

def plot_accuracy(accuracies):
    x = list(range(1, len(accuracies)+1))
    plt.plot(x, accuracies)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Accuracies')
    plt.title('Training Accuracy Per Epoch')
    plt.savefig('Training_Accuracies.png')

def save_model(model, dest):
    torch.save(model.state_dict(), dest)

def main():
    data_folder = 'here'
    if not os.path.exists(data_folder):
        dload.save_unzip("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip", # source url
                         data_folder, #destination folder
                         delete_after=True # dont keep zip afterwards
                         )
    training_data_path = os.path.join(data_folder, 'GTSRB')
    dest = './model.pth' # [folder]/[model].pth

    NUM_EPOCHS = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)
    model = Net() # instantiate model
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False) # optimizer
    criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean') # loss

    model.to(device)
    
    dataset = GTSRB('./GTSRB', transform=get_transform())
    train_loader = DataLoader(dataset, batch_size=50, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=3, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    accuracies = []
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}:'.format(epoch), end=' ')
        # training
        print('train', end=' ')
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        # evaluation
        print('eval')
        accuracy = evaluate(model, train_loader, device)
        accuracies.append(accuracy)

    save_model(model, dest)
    plot_accuracy(accuracies)

if __name__ == '__main__':
    main()