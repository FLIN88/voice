import torch
from dataset import simpledataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import os 
import torchvision
import pandas as pd
import svm
feature_type = 'MFCC'
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

print("loading model")
model = torchvision.models.resnet50(pretrained = True)
model.fc = torch.nn.Linear(2048, 2, bias = True)
model.load_state_dict(torch.load('./ckp/MFCC/best_ac_0.5948_0.0113.ckp'))

lossf = torch.nn.CrossEntropyLoss()


if not os.path.exists('ckp'):
    os.mkdir('ckp')

if torch.cuda.is_available():
    model = model.cuda()
    lossf = lossf.cuda()

#opt = torch.optim.SGD([{'params':[ param for name, param in model.named_parameters() if 'fc' in name]}], lr = lr, weight_decay = 1e-5)
opt = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = 0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma = 0.1)


def eval(data):
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
            
            loss += lossf(pre,y).cpu().item()
            ac += (pre.argmax(dim = 1) == y).sum().cpu().item()
    return ac / tot, loss / tot

def train_one(data):
    model.train()
    for x, y in data:
        x = x.cuda()
        y = y.cuda()
        pre = model(x)  
        loss = lossf(pre,y)

        opt.zero_grad()
        loss.backward()
        opt.step()

def train(usesvm = False):
    acls = []
    lossls = []
    bestac = 0
    bestloss = 0
    '''
    if usesvm:
        bestac = svm.eval(model)
        print("Epoch: {} -- AC: {:.2f}%".format(-1, bestac * 100))
    else:
        bestac, bestloss = eval(valid_loader)
        print("Epoch: {} -- AC: {:.2f}%, AVGloss: {:.4f}".format(-1, bestac * 100, bestloss))
    '''
    for ep in range(epoch):
        train_one(train_loader)
        scheduler.step()
        if usesvm is True:
            ac = svm.eval(model)
            acls.append(ac)
            if ac > bestac:
                bestac = ac
                torch.save(model.state_dict(), './ckp/best_svm_{:.4f}.ckp'.format(ac))
            print("Epoch: {} -- BestAC: {:.2f}%, AC: {:.2f}%".format(ep, bestac * 100, ac * 100))
        else:
            ac, loss = eval(valid_loader)
            acls.append(ac)
            lossls.append(loss)
            if ac > bestac:
                bestac = ac
                torch.save(model.state_dict(), './ckp/best_ac_{:.4f}_{:.4f}.ckp'.format(ac, loss))
            elif loss < bestloss:
                bestloss = loss
                torch.save(model.state_dict(), './ckp/best_ls_{:.4f}_{:.4f}.ckp'.format(ac, loss))
            print("Epoch: {} -- BestAC: {:.2f}%, AC: {:.2f}%, AVGloss: {:.4f}".format(ep, bestac * 100, ac * 100, loss))
    if usesvm:
        pd.DataFrame({'Accuracy': acls}).to_csv('./check.csv')
    else:
        pd.DataFrame({'Accuracy': acls, 'Loss': lossls}).to_csv('./check.csv')

train()    