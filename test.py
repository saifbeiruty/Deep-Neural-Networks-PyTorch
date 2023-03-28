import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

x = torch.randn(5, 5)
print(x)
x = x.view(-1,1,5,5)
print(x)