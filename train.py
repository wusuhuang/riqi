import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from util import Riqi, Logger
import argparse
import sys
import numpy as np
import os

def main(args):
    train_transform = T.Compose([T.ToTensor()])
    test_transform = T.Compose([T.ToTensor()])
    train_data = Riqi('/home/grid/project/riqi/data/train',transform=train_transform)
    test_data = Riqi("/home/grid/project/riqi/data/test", transform = test_transform)
    train_loder = DataLoader(train_data,batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loder = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)
    model = model.cuda()
    cretirion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    sys.stdout = Logger(args.path)


    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epoch):
        print("="*10 + f"train epoch: {epoch}"+ "="*10)
        model.train()
        for data, label in train_loder:
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            loss = cretirion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _,pred = torch.max(output, dim=-1)
            acc = (pred.cpu().numpy() == label.cpu().numpy()).astype(np.float32).mean()
            print(f"train loss: {loss:.3f}      train acc: {acc:.3f}")


        print("="*10 + f"evaluation at {epoch} epoch" + "="*10)
        model.eval()
        acc = 0.0
        num = 0.0
        for data, label in test_loder:
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            _,pred = torch.max(output, dim=-1)
            num += data.shape[0]
            acc += (pred.cpu().numpy() == label.cpu().numpy()).astype(np.float32).sum()

        print(f"test loss: {loss:.3f}      test acc: {acc/num:.3f}")
        
        if (acc/num) > best_acc:
            best_epoch = epoch
            best_acc = acc/num
            torch.save(model.state_dict(), args.path + "/model_best.pth")

        torch.save(model.state_dict(), args.path + "/model_latest.pth")
        print(f"best epoch:{best_epoch:.3f}  best_acc:{best_acc:.3f}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="forgery data detection")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--path", default="./result", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
