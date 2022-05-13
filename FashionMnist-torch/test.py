import argparse
import torch
import torchvision
from tqdm import tqdm
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--input_weights",type=str,help="Please Enter Your Weights")
parser.add_argument("--input_device",type=str,default="cpu",help="Please Enter Your Device -- cpu/cuda")
arg = parser.parse_args()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0),(1))
])

dataset_test = torchvision.datasets.FashionMNIST("./",train=False,download=True,transform=transform)
test_data = torch.utils.data.DataLoader(dataset_test,batch_size=32)

device = torch.device(arg.input_device)
model = Model()
model.load_state_dict(torch.load(arg.input_weights))
model.to(device)
model.eval()

class Test:
    def __init__(self):
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def accuracy(self,predict,label):
        _,pred_max = torch.max(predict,1)
        acc = torch.sum(pred_max==label,dtype=torch.float64) / len(predict)
        return acc

    def test_step(self,image,label):
        image.to(device)
        label.to(device)

        predict = model(image)
        loss = self.loss_function(predict,label)
        acc = self.accuracy(predict,label)

        return loss, acc
    
    def evaluate(self):
        test_loss = 0.0
        test_acc = 0.0

        for image, label in tqdm(test_data):
            loss, accuracy = self.test_step(image,label)
            test_loss += loss
            test_acc += accuracy

        total_loss = test_loss / len(test_data)
        total_acc = test_acc / len(test_data)

        print(f"Accuracy: {total_acc}, Loss: {total_loss}")

if __name__ == "__main__":
    mnist = Test()
    mnist.evaluate()
    