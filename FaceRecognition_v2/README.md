# Face Recognition

**Face Recognition Using YOLOv7 Algorithm And ArcFace Model**

![image1](https://user-images.githubusercontent.com/88143329/208358793-4937f2b3-d342-4dad-97c5-7953da775d5d.jpg)

 - Person Detection Using YOLOv7 Algorithm
 - Face Detection Using Mediapipe Model
 - Get Embedding Per Face Using ArcFace Model
 - Face Recognition Using Calculate Cosine Metrics From Embedds
 - Register App Using Pyside6


## installation
Clone repo and install requirements.txt
  ```
  git clone https://github.com/AlirezaKiaeipour/DeepLearning/tree/main/FaceRecognition_v2  # clone
  cd FaceRecognition_v2
  pip install -r requirements.txt  # install
  ```
  
## Register
```
$ python register.py
``` 
  
## Inference
Run the following command for ``Inference``
```
$ python suspect_recognition.py --weights [INPUT] --source [image|webcam] --image_path [INPUT|None]
``` 

## Reference
  
https://github.com/WongKinYiu/yolov7

https://github.com/serengil/deepface
