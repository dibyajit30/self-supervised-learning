import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
from collections import OrderedDict
import torchvision.transforms as T

def D(p, z):
  return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

class SimSiam(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50()
        self.backbone.fc=torch.nn.Identity()
        self.projector = nn.Sequential(
          nn.Linear(2048,2048),
          nn.BatchNorm1d(2048),
          nn.ReLU(inplace=True),
          nn.Linear(2048,2048),
          nn.BatchNorm1d(2048),
          nn.ReLU(inplace=True),
          nn.Linear(2048,2048),
          nn.BatchNorm1d(2048)
        )
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.predictor = nn.Sequential(
         nn.Linear(2048,512),
         nn.BatchNorm1d(512),
         nn.ReLU(inplace=True),
         nn.Linear(512,2048),
        )
    
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return L
    
def get_model():
    model=SimSiam()
    return model