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
from sklearn import svm

lr = 0.001
batch_size = 1024
epoch = 10

feature_type = 'MFCC'

train_set = simpledataset('train.csv', feature_type)
valid_set = simpledataset('valid.csv', feature_type)
test_set = simpledataset('test.csv', feature_type)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)


def loadmodel(path = None):
    model = torchvision.models.resnet50(pretrained = True)
    model.fc = torch.nn.Linear(2048, 2, bias = True)
    if torch.cuda.is_available():
        model = model.cuda()
    if path:
        model.load_state_dict(torch.load(path))
    return model
        


def feature(model, x):
    model.eval()
    x = x.cuda()
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
    return x.cpu()

def extract(model, loader):
    X, y = None, None
    for _X, _y in loader:
        pre = feature(model, _X)
        if X is None:
            X = pre
            y = _y
        else:
            X = torch.cat([X, pre])
            y = torch.cat([y, _y])
    return X, y

def eval(model):
    X, y = extract(model, train_loader)
    testX, testy = extract(model, valid_loader)
    clf = svm.SVC()
    clf.fit(X, y)
    return clf.score(testX, testy)

def evaldir(dir):
    for r, d, f in os.walk(dir):
        for name in f:
            path = os.path.join(r, name)
            print(name, eval(loadmodel(path)))

def test(model):
    X, y = extract(model, train_loader)
    testX, testy = extract(model, test_loader)
    clf = svm.SVC()
    clf.fit(X, y)
    return clf.score(testX, testy)

def testdir(dir):
    for r, d, f in os.walk(dir):
        for name in f:
            path = os.path.join(r, name)
            print(name, test(loadmodel(path)))

if __name__ == '__main__':
    print(eval(loadmodel()))
    print(test(loadmodel()))
    
    #evaldir('./ckp/logmel')