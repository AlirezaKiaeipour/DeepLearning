import argparse
import torch
from tqdm import tqdm
from model import tf_model
from dataset import Load_Data

parser = argparse.ArgumentParser()
parser.add_argument("--input_path",type=str,help="Please Enter Your Path")
parser.add_argument("--input_device",type=str,default="cpu",help="Please Enter Your Device -- cpu/cuda")
parser.add_argument("--input_epochs",type=int,default=20,help="Please Enter Number of Epochs")
arg = parser.parse_args()

dataset = Load_Data(arg.input_path)
train ,_ = dataset.get_data()
device = torch.device("cuda" if torch.cuda.is_available() and arg.input_device else "cpu")
model = tf_model()
model = model.to(device)

class Train:
    def __init__(self):
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        self.loss_function = torch.nn.MSELoss()
        self.epochs = arg.input_epochs

    def train_step(self,image,label):
        image = image.to(device)
        label = label.to(device)

        self.optimizer.zero_grad()
        predict = model(image)
        loss = self.loss_function(predict,label.float())
        loss.backward()
        self.optimizer.step()
  
        return loss

    def fit(self):
        model.train()
        for epoch in range(self.epochs):
            train_loss = 0.0
            for image, label in tqdm(train):
                loss = self.train_step(image,label)
                train_loss += loss
            
            total_loss = train_loss / len(train)

            print(f"Epoch: {epoch+1}, Loss: {total_loss}")

    def save_model(self):
        torch.save(model.state_dict(),"weights.pth")

if __name__ == "__main__":
    mnist = Train()
    mnist.fit()
    mnist.save_model()
    