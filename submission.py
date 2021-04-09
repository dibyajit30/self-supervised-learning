# Feel free to modifiy this file.

from torchvision import models, transforms

team_id = 16
team_name = "Integer"
email_address = "sg6148@nyu.edu"

def get_model():
    return models.resnet18(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])