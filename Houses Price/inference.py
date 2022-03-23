import argparse
import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_bedroom",type=int,help="Please Enter number of bedroom")
parser.add_argument("--input_bathroom",type=int,help="Please Enter number of bathroom")
parser.add_argument("--input_area",type=int,help="Please Enter number of area")
parser.add_argument("--input_zipcode",type=int,help="Please Enter number of zipcode")
parser.add_argument("--input_image_path",type=str,help="Please Enter path of input image")
arg = parser.parse_args()

model = load_model("weights/houseprice.h5")

with open("weights/StandardScaler","rb") as f:
    scaler = pickle.load(f)
with open("weights/LabelBinarizer","rb") as f:
    onehot = pickle.load(f)

info = np.array([[arg.input_bedroom,arg.input_bathroom,arg.input_area,arg.input_zipcode]])
scale = scaler.transform(info[:,:3])
label = onehot.transform(info[:,-1])
data = np.concatenate((scale,label),axis=1)

images = []
path = os.listdir(arg.input_image_path)
mask = np.zeros((64,64,3),np.uint8)
for img in path:
    img = cv2.imread(f"{arg.input_image_path}/{img}")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(32,32))
    images.append(img)
mask[:32,:32] = images[0]
mask[:32,32:] = images[1]
mask[32:,:32] = images[2]
mask[32:,32:] = images[3]
mask = np.expand_dims(mask,axis=0)
print(model.predict([data,mask]))