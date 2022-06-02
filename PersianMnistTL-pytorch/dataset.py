import torch
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = torchvision.datasets.ImageFolder(root="MNIST_persian",transform=transform)

def data():
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=4)

    return train , test