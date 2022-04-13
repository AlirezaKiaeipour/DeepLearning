import argparse
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
parser.add_argument("--input_image",type=str,help="Please Enter path of input image/  Example: image.jpg")
arg = parser.parse_args()

model = load_model(arg.input_model)

image = cv2.imread(arg.input_image)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(256,256)).astype(np.float32)
image = (image / 127.5) - 1
image = image[np.newaxis,...]

generate = model(image,training=True)
generate = np.squeeze(generate, axis=0)
generate = np.array((generate +1) *127.5).astype(np.uint8)
file_path= arg.input_image.split("/")
plt.imsave(f"output/{file_path[-1]}",generate)