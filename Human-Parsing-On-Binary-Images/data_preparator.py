import argparse
import os
import glob
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--path",type=str,help="Please Enter path of annotations dataset")
arg = parser.parse_args()

# Create Folder Dataset
os.makedirs("Dataset",exist_ok=True)
os.makedirs("Dataset/train_images",exist_ok=True)
os.makedirs("Dataset/val_images",exist_ok=True)
os.makedirs("Dataset/train_segmentations",exist_ok=True)
os.makedirs("Dataset/val_segmentations",exist_ok=True)

class Dataset:
    def __init__(self):
        super().__init__()
        self.path = glob.glob(os.path.join(f"{arg.path}/*"))
    
    def __call__(self):
        train_id = open("Dataset/train_id.txt", "w")
        val_id = open("Dataset/val_id.txt", "w")
        for index, img in enumerate(self.path):
            img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            img[img!=0] = 255
            if index <800:
                cv2.imwrite(f"Dataset/train_images/img{index}.jpg",img) # training binary person images with jpg format
            else:
                cv2.imwrite(f"Dataset/val_images/img{index}.jpg",img) # validation binary person images with jpg format
        
        for index, img in enumerate(self.path):
            img = cv2.imread(img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            # labels
            img[img==0]=0  # background
            img[img==38]=1  # head
            img[img==75]=2  # chest
            img[img==113]=3  # arms
            img[img==15]=4  # hands
            img[img==53]=5  # thigh
            img[img==90]=6  # legs & feets 
            if index <800:
                cv2.imwrite(f"Dataset/train_segmentations/img{index}.png",img) # training annotations grayscale person image with png format
                train_id.write(f"img{index}") # training image list
                train_id.write("\n")
            else:
                cv2.imwrite(f"Dataset/val_segmentations/img{index}.png",img) # validation annotations grayscale person image with png format
                val_id.write(f"img{index}") # validation image list
                val_id.write("\n")

        train_id.close()
        val_id.close()

if __name__ == '__main__':
    dataset = Dataset()
    dataset()
    