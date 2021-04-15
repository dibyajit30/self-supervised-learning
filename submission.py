# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict

team_id = 16
team_name = "Integer"
email_address = "sg6148@nyu.edu"

def get_classifier_model():
    model = models.resnet18(num_classes=800)
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.fc.in_features, 1000)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(1000, 1000)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(1000, 1000))
    ]))
    model.fc = classifier

    return model

class EncoderNet(nn.Module):
    def __init__(self, encoder_features=100, num_classes=800):
        super(EncoderNet, self).__init__()
        self.model = models.resnet18(num_classes=800)
    
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.model.fc.in_features, 1000)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(1000, 1000))
        ]))
        fc_layers = []
        fc_layers.append(classifier[:])
        fc_layers = nn.Sequential(*fc_layers)
        self.model.fc = fc_layers
        
    def forward(self, x):
        x = self.model(x)
        return(x)

class LinearNet(nn.Module):

    def __init__(self, encoder_features=1000, num_classes=800):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(encoder_features, encoder_features)
        self.fc2 = torch.nn.Linear(encoder_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return(x)
        
class CombinedNet(nn.Module):
    
    def __init__(self):
        super(CombinedNet, self).__init__()
        self.encoder = EncoderNet()
        self.classifier = LinearNet()
        
    def forward(self, x):
        features = self.encoder(x)
        output = self.classifier(features)
        return (output)
        
def get_model():
    model = CombinedNet()
    return model
        
def get_projection_head(model):
    fc_layers = []
    fc_layers.append(model.fc[:])
    fc_layers.append(nn.ReLU(inplace=True))
    fc_layers.append(nn.Linear(1000, 1000))
    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])
