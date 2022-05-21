import argparse
import time
import torch
import torchvision
import cv2
from google_drive_downloader import GoogleDriveDownloader as gdd
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--input_device",type=str,default="cpu",help="Please Enter Your Device -- cpu/cuda")
parser.add_argument("--input_image",type=str,help="Please Enter Your Image Path")
arg = parser.parse_args()

gdd.download_file_from_google_drive(file_id='1joypkcaMGNRUwpjsmqHW5IlZdq5aBq4e',dest_path='./weights/weights.pth')
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

device = torch.device("cuda" if torch.cuda.is_available() and arg.input_device else "cpu")
model = Model()
model.load_state_dict(torch.load("./weights/weights.pth"))
model.to(device)
model.eval()

start_time = time.time()
image = cv2.imread(arg.input_image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(70,70))

tensor = transform(image).unsqueeze(0).to(device)

predict = model(tensor)
end_time = time.time()
print(f"Predict: {predict} - Time: {end_time-start_time}")
