#!/usr/bin/env python
# coding: utf-8

# libraries setup
import torch
import torchvision

import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from glob import glob
from PIL import Image

# model
class Net(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        ...

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
        img_path,label = self.data[i]
        im = Image.open(img_path)
        if self.transform: im = self.transform(im)
        return im,label

    def __len__(self): return len(self.data)

def get_transform():
    return T.Compose([T.Resize((48,48)), T.ToTensor(), T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])])

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
    torch.manual_seed(0)

    training_data_path = ...
    NUM_EPOCHS = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = ... # optimizer
    criterion = ... # loss
    model = Net(...) # instantiate model
    dest = ... # [folder]/[model].pth

    model.to(device)
    
    dataset = GTSRB('./GTSRB',transform=get_transform())
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    accuracies = []
    for epoch in range(NUM_EPOCHS):
        # training
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        # evaluation
        accuracy = evaluate(model, train_loader, device)
        accuracies.append(accuracy)

    save_model(model, dest)
    plot_accuracy(accuracies)

# test
# plot_accuracy(np.random.normal(1,2,50))