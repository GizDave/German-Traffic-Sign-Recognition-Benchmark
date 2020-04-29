'''
Citation:

The implementation of the following Net is based on the the Conv net architecture mentioned in the following paper:
Deep Neural Network for Traffic Sign Recognition Systems: An analysis of Spatial Transformers and Stochastic Optimization Methods
(Alvaro Arcos-Garc ́ıa, Juan A. Alvarez-Garc ́ıa, Luis M. Soria-Morillo)
Paper Link: https://idus.us.es/bitstream/handle/11441/80679/NEUNET-D-17-00381.pdf

The local contrast normalisation with Gaussians kernels (LCN) is from the following paper:
Local Contrast Normalization
(Kevin Jarrett, Koray Kavukcuoglu, Marc’Aurelio Ranzato and Yann LeCun)
Paper Link: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf

The implementation of the  is based on the following page (with slight modifications & bug fix):
https://github.com/dibyadas/Visualize-Normalizations/blob/master/LocalContrastNorm.ipynb

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
