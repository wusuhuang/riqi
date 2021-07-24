import sys
from typing import SupportsRound
import cv2
import pdb
import shutil
import os

path = "/home/grid/project/riqi/data"


# for label in os.listdir(path):
#     for filename in os.listdir(os.path.join(path, label)):
#         name = filename[2:]
        
#         flag = 0 if 'negtive' in label else 1
#         shutil.move(os.path.join(os.path.join(path,label),filename), os.path.join(os.path.join(path, label), f'{flag}_'+name))


# split the train and test data for making daraset
train_path = "/home/grid/project/riqi/data/train"
test_path = "/home/grid/project/riqi/data/test"
source_p = "/home/grid/project/riqi/data/positive-1"
source_n = "/home/grid/project/riqi/data/negtive-1"

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
p = os.listdir(source_p)
n = os.listdir(source_n)

for filename in p[:int(0.8*len(p))]:
    shutil.copy(os.path.join(source_p,filename), os.path.join(train_path, filename))
for filename in n[:int(0.8*len(n))]:
    shutil.copy(os.path.join(source_n,filename), os.path.join(train_path, filename))
for filename in p[int(0.8*len(p)):]:
    shutil.copy(os.path.join(source_p,filename), os.path.join(test_path, filename))
for filename in n[int(0.8*len(n)):]:
    shutil.copy(os.path.join(source_n,filename), os.path.join(test_path, filename))

