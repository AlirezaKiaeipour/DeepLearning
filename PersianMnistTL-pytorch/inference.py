import argparse
import time
import torch
import torchvision
import cv2
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
from model import tf_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_weights",type=str,help="Please Enter Your Weights Path")
parser.add_argument("--input_device",type=str,default="cpu",help="Please Enter Your Device -- cpu/cuda")
parser.add_argument("--input_image",type=str,help="Please Enter Your Image Path")
arg = parser.parse_args()

# gdd.download_file_from_google_drive(file_id='15zd5_zM_xmqlw3EocyRPNQNpZ4vUM38q',dest_path='./weights/weights.pth')
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

device = torch.device("cuda" if torch.cuda.is_available() and arg.input_device else "cpu")
model = tf_model()
model.load_state_dict(torch.load(arg.input_weights))
model.to(device)
model.eval()

start_time = time.time()
image = cv2.imread(arg.input_image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(224,224))

tensor = transform(image).unsqueeze(0).to(device)

predict = model(tensor)
predict = predict.cpu().detach().numpy()
end_time = time.time()
print(f"Predict: {np.argmax(predict)} - Time: {end_time-start_time}")
