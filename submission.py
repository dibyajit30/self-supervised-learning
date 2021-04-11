# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict

team_id = 16
team_name = "Integer"
email_address = "sg6148@nyu.edu"

def get_model():
    model = models.resnet18(num_classes=800)
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.fc.in_features, 100)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(100, 100)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(100, 100))
    ]))
    model.fc = classifier

    return model

class LinearNet(nn.Module):

    def __init__(self, encoder_features=100, num_classes=800):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(encoder_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return(x)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])