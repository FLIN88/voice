import torch
import os
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from torchvision import transforms



features = {
    'spec': '.\\spec',
    'logmel': '.\\logmel',
    'MFCC': '.\\MFCC'
}

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])

class simpledataset(Dataset):
    def __init__(self, csv_path, featrue_type):
        self.data = pd.read_csv(csv_path)
        self.type = featrue_type
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret = []
        label = self.data.iloc[idx]['label']
        path = self.data.iloc[idx]['path']
        f = open(os.path.join(features[self.type], path), 'rb')
        t = torch.from_numpy(pickle.load(f)).to(torch.float32).unsqueeze(0)
        f.close()
        t = resize(t, (224, 224)).squeeze()
        t -= t.min()
        t /= t.max()
        return normalize(torch.stack([t, t, t])), label



