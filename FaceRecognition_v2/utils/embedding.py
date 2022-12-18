import json
import numpy as np
import cv2
import models.ArcFace as ArcFace
from scipy.spatial import distance

# Load ArcFace Model
model = ArcFace.loadModel()

def prerpocess(img):
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image,(112,112))  # Resize to 112*112
    img_pred = image / 255  # Normalize image [0,1]
    img_pred = np.expand_dims(img_pred,axis=0)
    return img_pred

# calculate embedd image using ArcFace Model (1,512)
def calc_embedd(img):
    return model.predict(img)

# face recognition using calculate cosine metrics from present embedded and saved embedded in json file
def identity(img,json_file="data/database.json"):
    threshhold = 0.68
    image = prerpocess(img)
    embedd = calc_embedd(image)
    with open(json_file) as f:
        data = json.load(f)
        for index,i in enumerate(data["information"]):
            if distance.cosine(embedd[0],list(i.values())[4]) <= threshhold: return f"{list(i.values())[0]} {list(i.values())[1]}"
        return 'Not identified'
