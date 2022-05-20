import argparse
import torch
from tqdm import tqdm
from model import Model
from data import data

parser = argparse.ArgumentParser()
parser.add_argument("--input_device",type=str,default="cpu",help="Please Enter Your Device -- cpu/cuda")
parser.add_argument("--input_epochs",type=int,default=20,help="Please Enter Number of Epochs")
arg = parser.parse_args()

train , _ = data()
device = torch.device("cuda" if torch.cuda.is_available() and arg.input_device else "cpu")
model = Model()
model = model.to(device)
model.train = True

class Train:
    def __init__(self):
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.epochs = arg.input_epochs

    def accuracy(self,predict,label):
        _,pred_max = torch.max(predict,1)
        acc = torch.sum(pred_max==label,dtype=torch.float64) / len(predict)
        return acc

    def train_step(self,image,label):
        image.to(device)
        label.to(device)

        self.optimizer.zero_grad()
        predict = model(image)
        loss = self.loss_function(predict,label)
        loss.backward()
        self.optimizer.step()
        accuracy = self.accuracy(predict,label)

        return loss , accuracy

    def fit(self):
        for epoch in range(self.epochs):
            train_loss = 0.0
            train_acc = 0.0

            for image, label in tqdm(train):
                loss , accuracy = self.train_step(image,label)
                train_loss += loss
                train_acc += accuracy
            
            total_loss = train_loss / len(train)
            total_acc = train_acc / len(train)

            print(f"Epoch: {epoch+1}, Accuracy: {total_acc}, Loss: {total_loss}")

    def save_model(self):
        torch.save(model.state_dict(),"weights.pth")

if __name__ == "__main__":
    mnist = Train()
    mnist.fit()
    mnist.save_model()
    