import argparse
import cv2
import numpy as np
from model import Mymodel

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
parser.add_argument("--input_image",type=str,help="Please Enter path of input image/  Example: image.jpg")
arg = parser.parse_args()

model = Mymodel()
model.load_weights(arg.input_model)

img = cv2.imread(arg.input_image)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(224,224))
img = img / 255.0
img = np.expand_dims(img,axis=0)
predict = np.argmax(model.predict(img))
person = ["Ali Khamenei","Angelina Jolie","Barak Obama","Behnam Bani","Donald Trump","Emma Watson",
    "Han Hye Jin","Kim Jong Un","Leyla Hatami","Lionel Messi","Michelle Obama","Morgan Freeman",
    "Queen Elizabeth","Scarlett Johansson"]
print(person[predict])
