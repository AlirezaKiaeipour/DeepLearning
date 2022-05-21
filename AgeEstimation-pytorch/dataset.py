import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((70,70)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

class CustomDataset(Dataset):
    def __init__(self,image,label,transform):
        self.images = image
        self.labels = label
        self.transform = transform
    
    def __getitem__(self,index):
        labels = self.labels[index]
        images = self.images[index]
        images = self.transform(images)

        return images, labels
            
    def __len__(self):
        return len(self.labels)


class Load_Data:
    def __init__(self,path):
        self.path = path

    def load_dataset(self):
        images = []
        ages = []
        for image in os.listdir(self.path):
            age = int(image.split("_")[0])
            ages.append(age)
            img = cv2.imread(f"{self.path}/{image}")
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            images.append(img)

        images = pd.Series(images,name="Images")
        ages = pd.Series(ages,name="Ages")
        df = pd.concat([images,ages],axis=1)
        under_4 = df[df["Ages"]<=4]
        under_4_new = under_4.sample(frac=0.3)
        up_4 = df[df["Ages"]>4]
        df = pd.concat([under_4_new,up_4],axis=0)
        df = df[df["Ages"]<70]

        X = np.array(df["Images"].tolist())
        Y = np.array(df["Ages"].tolist())

        return X, Y

    def get_data(self):
        image,label = self.load_dataset()
        dataset = CustomDataset(image,label,transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train, test
