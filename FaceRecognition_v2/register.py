import sys
import json
import numpy as np
import cv2
import mediapipe as mp
import models.ArcFace as ArcFace
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage,QPixmap

# Register person in application 
class Register(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load ArcFace Model
        self.model = ArcFace.loadModel()
        loader = QUiLoader()

        # Load UI
        self.ui = loader.load("ui/form_cam.ui",None)
        self.ui.show()
        self.ui.added.clicked.connect(self.save)
        self.ui.actionInfo.triggered.connect(self.info)
        self.ui.actionExit.triggered.connect(exit)

        # Load mediapipe Model
        mpfacedetection = mp.solutions.face_detection
        detector = mpfacedetection.FaceDetection()
        cap = cv2.VideoCapture(0)
        while True:
            _,self.frame = cap.read()
            self.frame_rgb = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            self.frame_rgb = cv2.resize(self.frame_rgb,(480,360))  # Resize frame to 480*360
            faces = detector.process(self.frame_rgb)
            if faces.detections:
                for index,detection in enumerate(faces.detections):
                    bounding_box = detection.location_data.relative_bounding_box
                    height,width = self.frame_rgb.shape[:2]
                    x ,y= int(bounding_box.xmin*width) ,int(bounding_box.ymin*height)
                    w ,h = int(bounding_box.width*width) ,int(bounding_box.height*height)
                    self.face = self.frame_rgb[y-20:y+h,x:x+w]
                    try:
                        cv2.rectangle(self.frame,(x,y-20),(x+w,y+h),(0,255,0),2)
                    except:
                        pass
            
            # show frame of video on application 
            img = QImage(self.frame_rgb, self.frame_rgb.shape[1], self.frame_rgb.shape[0],QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.ui.label_img.setPixmap(pixmap)
            self.key = cv2.waitKey(1)

    def write_json(self,data,filename="data/database.json"):
        with open(filename,"w") as f:
            json.dump(data,f,indent=4)
        
    # save information include: First Name, Last Name, Nation Code , Phone Number and Image Embed Per Face
    def save(self):
        first_name = self.ui.first_name.text()
        last_name = self.ui.last_name.text()
        n_code = self.ui.nation_code.text()
        phone_number = self.ui.phone.text()
        if first_name=="" or last_name=="" or n_code=="" or phone_number=="":
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("Please complete the fields")
            msg.setIcon(QMessageBox.Critical)
            msg.exec()
        else:
            self.frame_face = cv2.resize(self.face,(112,112))  # Resize image to 112*112
            self.frame_face = self.frame_face / 255  # Normalize image[0,1]
            self.frame_face = np.expand_dims(self.frame_face,axis=0)
            img_embed = self.model.predict(self.frame_face)  # Get embedding per face
            
            # save information in json file
            with open("data/database.json") as json_file:
                data = json.load(json_file)
                temp = data["information"]
                info = {"first_name":first_name,
                "last_name":last_name,
                "national_code":n_code,
                "phone_number":phone_number,
                "img_embed":img_embed[0].tolist()}
                temp.append(info)
            
            self.write_json(data)
            self.ui.first_name.setText("")
            self.ui.last_name.setText("")
            self.ui.nation_code.setText("")
            self.ui.phone.setText("")
    
    # about developer
    def info(self):
        msg = QMessageBox()
        msg.setWindowTitle("Info")
        msg.setText("Face Recognition")
        msg.setInformativeText("GUI Face Recognition using YOLOv7, ArcFace and Pyside6\nThis program was developed by Alireza Kiaeipour\nContact developer: a.kiaipoor@gmail.com\nBuilt in 2022")
        msg.setIcon(QMessageBox.Information)
        msg.exec()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Register()
    app.exec()
