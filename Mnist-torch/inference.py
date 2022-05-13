import argparse
import time
import torch
import torchvision
import cv2
import numpy as np
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--input_weights",type=str,help="Please Enter Your Weights")
parser.add_argument("--input_device",type=str,default="cpu",help="Please Enter Your Device -- cpu/cuda")
parser.add_argument("--input_image",type=str,help="Please Enter Your Image Path")
arg = parser.parse_args()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0),(1))
])

device = torch.device(arg.input_device)
model = Model()
model.load_state_dict(torch.load(arg.input_weights))
model.to(device)
model.eval()

start_time = time.time()
image = cv2.imread(arg.input_image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = cv2.resize(image,(28,28))

tensor = transform(image).unsqueeze(0)
tensor = tensor.to(device)

predict = model(tensor)
predict = predict.cpu().detach().numpy()
end_time = time.time()
print(f"Predict: {np.argmax(predict)} - Time: {end_time-start_time}")
