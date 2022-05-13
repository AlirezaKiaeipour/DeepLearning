import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fully_connect1 = torch.nn.Linear(784,512)
        self.fully_connect2 = torch.nn.Linear(512,256)
        self.fully_connect3 = torch.nn.Linear(256,10)
    
    def forward(self,x):
        x = x.reshape((x.shape[0],784))
        x = self.fully_connect1(x)
        x = torch.relu(x)
        x = torch.dropout(x,0.2,train=True)
        x = self.fully_connect2(x)
        x = torch.relu(x)
        x = torch.dropout(x,0.5,train=True)
        x = self.fully_connect3(x)
        x = torch.softmax(x,dim=1)

        return x
        