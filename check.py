import pandas as pd
import torch
import numpy as np
import pickle
from torchvision import transforms
from torch.nn.functional import adaptive_max_pool2d
from torchvision.transforms.functional import resize
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import simpledataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import os

feature_type = 'MFCC'
train_set = simpledataset('train.csv', feature_type)
valid_set = simpledataset('valid.csv', feature_type)
test_set = simpledataset('test.csv', feature_type)

lr = 0.001
batch_size = 64
epoch = 10

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
print("datasets loaded")

def eval(data, model, lossf):
    ac = 0
    tot = 0
    loss = 0.
    model.eval()
    with torch.no_grad():
        for x, y in data:
            tot += len(y)
            x = x.cuda()
            y = y.cuda()
            pre = model(x)
            
            loss += lossf(pre,y).cpu()
            ac += (pre.argmax(dim = 1) == y).sum().cpu().item()
    return ac / tot, loss / tot

def check(path):
    model = torchvision.models.resnet50(pretrained = True)
    model.fc = torch.nn.Linear(2048, 2, bias = True)
    lossf = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        lossf = lossf.cuda()

    model.load_state_dict(torch.load(path))
    ac, loss = eval(test_loader, model, lossf)

    print(path, "AC: {:.2f}%, AVGloss: {:.4f}".format(ac * 100, loss))
'''
for r, d, f in os.walk('./ckp/spec'):
    for name in f:
        if name.endswith('.ckp'):
            check(os.path.join(r, name))   
'''

check('./ckp/spec/best_ac_0.5172_0.0115.ckp')