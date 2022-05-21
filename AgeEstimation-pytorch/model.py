import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,(3,3),(1,1),(1,1))
        self.conv2 = nn.Conv2d(64,128,(3,3),(1,1),(1,1))
        self.conv3 = nn.Conv2d(128,256,(3,3),(1,1),(1,1))
        
        self.fully_connect1 = nn.Linear(256*8*8,256)
        self.fully_connect2 = nn.Linear(256,128)
        self.fully_connect3 = nn.Linear(128,1)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,kernel_size=(2,2))
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.fully_connect1(x))
        x = torch.dropout(x,0.2,train=True)
        x = F.relu(self.fully_connect2(x))
        x = torch.dropout(x,0.5,train=True)
        x = self.fully_connect3(x)
        
        return x