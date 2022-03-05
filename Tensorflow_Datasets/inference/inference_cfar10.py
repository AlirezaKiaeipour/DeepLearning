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
image = cv2.resize(image,(32,32))
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = image / 255.0
image = image.reshape(1,32,32,3)

predict = np.argmax(model.predict(image))
if predict == 0:
    print("Airplane")
elif predict == 1:
    print("Automobile")
elif predict == 2:
    print("Bird")
elif predict == 3:
    print("Cat")
elif predict == 4:
    print("Deer")
elif predict == 5:
    print("Dog")
elif predict == 6:
    print("Frog")
elif predict == 7:
    print("Horse")
elif predict == 8:
    print("Ship")
elif predict == 9:
    print("Truck")