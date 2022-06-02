import torch.nn as nn
import torchvision.models as models

def tf_model():
    model = models.resnet152(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features,10)
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False

    return model
    