import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
parser.add_argument("--input_image",type=str,help="Please Enter path of input image/  Example: image.jpg")
arg = parser.parse_args()

model = load_model(arg.input_model)

image = cv2.imread(arg.input_image)
image = cv2.resize(image,(64,64))
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = image / 255.0
image = np.expand_dims(image,axis=0)

predict = np.argmax(model.predict(image))
print(predict)
