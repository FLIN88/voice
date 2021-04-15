import torch
from dataset import simpledataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import os 
import torchvision
import pandas as pd
import svm
from torchvision.transforms.functional import resize
from torchvision import transforms
import pickle
feature_type = ['spec', 'logmel', 'MFCC']
modeldir = {
    'spec': './ckp/spec/best_svm_0.6724.ckp',
    'logmel': './ckp/logmel/best_bls_0.6034_0.0144.ckp'
}

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])

train_set = simpledataset('train.csv', feature_type)
valid_set = simpledataset('valid.csv', feature_type)
test_set = simpledataset('test.csv', feature_type)

lr = 0.001
batch_size = 64
epoch = 50

print("loading datasets")
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)

prefix = './features'

if not os.path.exists(prefix):
    os.mkdir(prefix)

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
    return x.cpu().squeeze()

def readpk(path):
    f = open(path, 'rb')
    t = torch.from_numpy(pickle.load(f)).to(torch.float32).unsqueeze(0)
    f.close()
    t = resize(t, (224, 224)).squeeze()
    t -= t.min()
    t /= t.max()
    return normalize(torch.stack([t, t, t])).unsqueeze(0)

def savefeature(data, path):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__':
    for ft in feature_type:
        if not os.path.exists(os.path.join(prefix, ft)):
            os.mkdir(os.path.join(prefix, ft))
        model = torchvision.models.resnet50(pretrained = True)
        model.fc = torch.nn.Linear(2048, 2, bias = True)
        if ft != 'MFCC':
            model.load_state_dict(torch.load(modeldir[ft]))
        model.cuda()
        for r, d, f in os.walk(os.path.join('./', ft)):
            if 'git' in r:
                continue 
            print(ft, r)
            for name in f:
                if name.endswith('.pk'):
                    x = readpk(os.path.join(r, name))
                    x = feature(model, x).numpy()
                    savefeature(x, os.path.join(prefix, r[2:], name))
            for name in d:
                if not name.startswith('.') and not os.path.exists(os.path.join(prefix, r[2:], name)):
                    os.mkdir(os.path.join(prefix, r[2:], name))
                    
            
