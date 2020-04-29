import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from glob import glob
from PIL import Image

class tsDataset(data.dataset):
	def __init__(self, path=):
		self.lbl, self.img = [], []

		self.img_transform = transforms.Compose([
			...
		])

	def __getitem__(self, index):
		img = self.img_transform(self.img[index])
		return img, self.lbl[index]

	def __len__(self):
		return len(self.img)


