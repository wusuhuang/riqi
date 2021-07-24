from typing import NoReturn
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2
import sys
import pdb


class Riqi(Dataset):
    def __init__(self, path,transform=None):
        super().__init__()
        self.path = path
        self.img_list = os.listdir(path)
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = cv2.imread(os.path.join(self.path, img_path))
        img = cv2.resize(img, (224,224))
        label = int(img_path[0])
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_list)

class Logger(object):
    def __init__(self, path):
        self.console = sys.stdout
        self.file = None
        self.path = os.path.join(path,"train.log")
        if not os.path.exists(path):
            os.makedirs(path)

        # pdb.set_trace()
        self.file = open(self.path, "w")

    def __del__(self):
        self.close()
    def __exter__(self):
        pass
    def __exit__(self, *args):
        self.close()
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()