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
image = cv2.resize(image,(28,28))
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = image / 255.0
image = image.reshape(1,28,28,1)

predict = np.argmax(model.predict(image))
if predict == 0:
    print("T-shirt")
elif predict == 1:
    print("Trouser")
elif predict == 2:
    print("Pullover")
elif predict == 3:
    print("Dress")
elif predict == 4:
    print("Coat")
elif predict == 5:
    print("Sandal")
elif predict == 6:
    print("Shirt")
elif predict == 7:
    print("Sneaker")
elif predict == 8:
    print("Bag")
elif predict == 9:
    print("Ankle boot")