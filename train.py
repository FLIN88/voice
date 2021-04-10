import torch
from dataset import simpledataset
from torch.utils.data import DataLoader
import torchvision.models

logmel_train_set = simpledataset('train.csv', 'logmel')
logmel_valid_set = simpledataset('valid.csv', 'logmel')
logmel_test_set = simpledataset('test.csv', 'logmel')

lr = 1e-2
batch_size = 64

logmel_train_loader = DataLoader(logmel_train_set, batch_size = batch_size, shuffle = True)
logmel_valid_loader = DataLoader(logmel_valid_set, batch_size = batch_size, shuffle = True)
logmel_test_loader = DataLoader(logmel_test_set, batch_size = batch_size, shuffle = True)



model = torchvision.models.resnet50(pretrained = True)

model.fc = torch.nn.Linear(2048, 2, bias = True)
model.eval()

for x, y in logmel_valid_loader:
    pre = model(x)
    print(pre)

lossf = torch.nn.CrossEntropyLoss()

