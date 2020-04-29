import os
import torch.utils.data as data
from glob import glob
from PIL import Image,ImageDraw
import torchvision.transforms as T

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

    def get_orig_img_path(self,i): return self.data[i]

def get_transform():
    return T.Compose([
        T.Resize((48,48)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])
