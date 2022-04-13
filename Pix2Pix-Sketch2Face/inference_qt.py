import sys
import argparse
from PySide6.QtWidgets import QMainWindow , QApplication , QMessageBox,QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage,QPixmap
import cv2
import numpy as np
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
arg = parser.parse_args()

class Sketch(QMainWindow):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.ui = loader.load("form.ui",None)
        self.ui.show()
        self.model = load_model(arg.input_model)
        self.ui.btn_image.clicked.connect(self.openfile)
        self.ui.btn_pix.clicked.connect(self.pix2pix)

    def openfile(self):
        save=QFileDialog.getOpenFileName(self,caption="Open File as",dir=".",filter="All Files (*.*)")
        img = cv2.imread(save[0])
        self.img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_rgb = QImage(self.img_rgb, self.img_rgb.shape[1], self.img_rgb.shape[0],QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img_rgb)
        self.ui.img.setPixmap(pixmap)

    def pix2pix(self):
        image = cv2.resize(self.img_rgb,(256,256)).astype(np.float32)
        image = image[np.newaxis,...]
        image = (image / 127.5) - 1

        generate = self.model(image,training=True)
        generate = np.squeeze(generate, axis=0)
        generate = np.array((generate +1) * 127.5).astype(np.uint8)
        img = QImage(generate, generate.shape[1], generate.shape[0],QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.ui.img1.setPixmap(pixmap)     

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Sketch()
    app.exec()