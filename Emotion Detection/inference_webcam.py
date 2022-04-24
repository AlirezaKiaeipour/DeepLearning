import argparse
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
arg = parser.parse_args()

model = load_model(arg.input_model)
mpfacedetection = mp.solutions.face_detection
detector = mpfacedetection.FaceDetection()
cap = cv2.VideoCapture(0)
while True:
  _,frame = cap.read()
  frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  face = detector.process(frame_rgb)
  if face.detections:  
    for index,detection in enumerate(face.detections):
      bounding_box = detection.location_data.relative_bounding_box
      height,width = frame.shape[:2]
      x ,y= int(bounding_box.xmin*width) ,int(bounding_box.ymin*height)
      w ,h = int(bounding_box.width*width) ,int(bounding_box.height*height)

      pred_frame = frame_rgb[y-30:y+h,x:x+w]
      pred_frame = cv2.cvtColor(pred_frame,cv2.COLOR_BGR2GRAY)
      try:
        pred_frame = cv2.resize(pred_frame,(48,48))
        pred_frame = pred_frame / 255.0
        pred_frame = pred_frame.reshape(1,48,48,1)

        predict = np.argmax(model.predict(pred_frame))
        if predict==0:predict="Anger"
        elif predict==1:predict="Disgust"
        elif predict==2:predict="Fear"
        elif predict==3:predict="Happy"
        elif predict==4:predict="Neutral"
        elif predict==5:predict="Sad"
        elif predict==6:predict="Surprise"
        
        cv2.rectangle(frame,(x,y-30),(x+w,y+h),(0,255,0),4)
        cv2.putText(frame,str(predict),(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
      except:
        pass
  cv2.imshow("Emotion Detection",frame)
  if cv2.waitKey(1) == 27:
    break

cap.release()
cv2.destroyAllWindows()