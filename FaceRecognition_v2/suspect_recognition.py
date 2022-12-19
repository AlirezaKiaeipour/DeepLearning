import argparse
import torch
import cv2
import numpy as np
import mediapipe as mp
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.embedding import identity

class SuspectRecognition:
    def __init__(self):
        weights = arg.weights
        device = 'cpu'
        image_size = 640
        trace = True

        # Initialize
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(image_size, s=stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, image_size)

        if self.half:
            self.model.half()  # to FP16
            
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def personDetection(self,source_image):
        # Padded resize
        img_size = 640
        stride = 32
        img = letterbox(source_image, img_size, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.array(img,dtype="uint8")
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        with torch.no_grad():
            # Inference
            pred = self.model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)
        person_detections = []
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()

                # Return results
                for *xyxy, conf, cls in reversed(det):
                    coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                    person_detections.append(coords)

        return person_detections


    # Face Recognition From Image
    def faceRecognition_from_image(self,image):
        if image is None:
            return None
        image = cv2.imread(arg.image_path)
        person_detections = self.personDetection(image)  # person detection per image
        for xywh in person_detections:
            plot_one_box(xywh, image, label="Person", color=[0, 0, 255], line_thickness=1)  # draw rectangle per person

        # Load mediapipe model
        mpfacedetection = mp.solutions.face_detection
        detector = mpfacedetection.FaceDetection()
        frame_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        faces = detector.process(frame_rgb)
        if faces.detections:  
            for index,detection in enumerate(faces.detections):
                bounding_box = detection.location_data.relative_bounding_box
                height,width = image.shape[:2]
                x ,y= int(bounding_box.xmin*width) ,int(bounding_box.ymin*height)
                w ,h = int(bounding_box.width*width) ,int(bounding_box.height*height) 
                face = image[y-20:y+h,x:x+w]
                plot_one_box([x,y-20,x+w,y+h], image, label=identity(face), color=[0, 255, 0], line_thickness=1)  # draw rectangle and face recognition using identity function

        cv2.imshow("Face Recognition",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows


    # Face Recognition From Webcam
    def faceRecognition_from_webcam(self):
        # Load mediapipe model
        mpfacedetection = mp.solutions.face_detection
        detector = mpfacedetection.FaceDetection()
        cap = cv2.VideoCapture(0)
        while True:
            _,frame = cap.read()
            person_detections = self.personDetection(frame)  # person detection per frame
            for xywh in person_detections:
                plot_one_box(xywh, frame, label="Person", color=[0, 0, 255], line_thickness=1)  # draw rectangle per person

            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            faces = detector.process(frame_rgb)
            if faces.detections:  
                for index,detection in enumerate(faces.detections):
                    bounding_box = detection.location_data.relative_bounding_box
                    height,width = frame.shape[:2]
                    x ,y= int(bounding_box.xmin*width) ,int(bounding_box.ymin*height)
                    w ,h = int(bounding_box.width*width) ,int(bounding_box.height*height) 
                    face = frame[y-20:y+h,x:x+w]
                    try:
                        plot_one_box([x,y-20,x+w,y+h], frame, label=identity(face), color=[0, 255, 0], line_thickness=1)  # draw rectangle and face recognition using identity function
                    except:
                        pass

            cv2.imshow("Face Recognition",frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",type=str,default='weights/yolov7-tiny.pt',help='model.pt')
    parser.add_argument("--source",type=str,default='webcam',help='webcam OR image')
    parser.add_argument("--image_path",type=str,default=None,help='Image Path')
    arg = parser.parse_args()
    faceRecognition = SuspectRecognition()
    if arg.source == "image":
        faceRecognition.faceRecognition_from_image(arg.image_path)
    if arg.source == "webcam":
        faceRecognition.faceRecognition_from_webcam()
