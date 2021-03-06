# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict

team_id = 16
team_name = "Integer"
email_address = "sg6148@nyu.edu"


class EncoderNet(nn.Module):
    def __init__(self, encoder_features=100, num_classes=800):
        super(EncoderNet, self).__init__()
        self.model = models.resnet50()
    
        '''classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.model.fc.in_features, 1000)),
            ('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(1000, 1000))
        ]))
        
        self.model.fc = classifier'''
        
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
        
def get_projection_head(model, num_classes=800):
    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model.fc.in_features, model.fc.out_features)),
            #('added_relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(model.fc.out_features, model.fc.out_features)),
            ('added_relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(model.fc.out_features, num_classes)),
        ]))
    return classifier

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])
