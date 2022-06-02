import torchvision
import torch.nn as nn

def tf_model():
    model = torchvision.models.resnet152(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features,1)
    ct = 0
    for child in model.children():
        ct +=1
        if ct<7:
            for param in child.parameters():
                param.requires_grad = False

    return model
