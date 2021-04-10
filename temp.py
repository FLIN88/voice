import pandas as pd
import torch
import pickle
from torchvision import transforms
from torch.nn.functional import adaptive_max_pool2d
from torchvision.transforms.functional import resize
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import simpledataset
from torch.utils.data import DataLoader


x = torch.rand((2, 3))
x = x.double()
print(x)