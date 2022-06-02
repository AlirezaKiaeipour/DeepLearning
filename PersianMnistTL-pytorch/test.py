import argparse
import torch
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd
from model import tf_model
from dataset import data

parser = argparse.ArgumentParser()
parser.add_argument("--input_weights",type=str,help="Please Enter Your Weights Path")
parser.add_argument("--input_device",type=str,default="cpu",help="Please Enter Your Device -- cpu/cuda")
arg = parser.parse_args()
# gdd.download_file_from_google_drive(file_id='15zd5_zM_xmqlw3EocyRPNQNpZ4vUM38q',dest_path='./weights/weights.pth')


_,test = data()
device = torch.device("cuda" if torch.cuda.is_available() and arg.input_device else "cpu")
model = tf_model()
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
        image = image.to(device)
        label = label.to(device)

        predict = model(image)
        loss = self.loss_function(predict,label)
        acc = self.accuracy(predict,label)

        return loss, acc
    
    def evaluate(self):
        test_loss = 0.0
        test_acc = 0.0

        for image, label in tqdm(test):
            loss, accuracy = self.test_step(image,label)
            test_loss += loss
            test_acc += accuracy

        total_loss = test_loss / len(test)
        total_acc = test_acc / len(test)

        print(f"Accuracy: {total_acc}, Loss: {total_loss}")

if __name__ == "__main__":
    mnist = Test()
    mnist.evaluate()
    