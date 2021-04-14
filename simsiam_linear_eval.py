import os
import argparse

from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from collections import OrderedDict
import torchvision.transforms as T
from torchvision import datasets
from dataloader import CustomDataset
from submission_simsiam import get_encoder, D, get_classifier
from transform import generate_pairs_simsiam
import time

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()
encoder_checkpoint_path = os.path.join(args.checkpoint_path, "simsiam_encoder.pth")
trainset = CustomDataset(root='/dataset', split="train", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
checkpoint=torch.load(encoder_checkpoint_path)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model=get_encoder()
model.load_state_dict(checkpoint)
model=torch.nn.DataParallel(model)
model=model.to(device)

classifier = get_classifier()
classifier=torch.nn.DataParallel(classifier)
classifier=classifier.to(device)

optimizer = torch.optim.SGD(classifier.parameters(), lr=30, momentum=0.9)

batch=1
print("Started Supervised training")
model=model.module.backbone.to(device)
for epoch in range(10):
        model.eval()
        classifier.train()
        running_loss = 0.0
        for idx, (images, labels) in enumerate(trainloader):    
            with torch.no_grad():
                feature = model(images.to(device))
            preds = classifier(feature)
            loss = F.cross_entropy(preds, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        if (idx) % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' % (epoch + 1, idx+1, running_loss / 10))
            running_loss = 0.0            
        batch+=1
os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(classifier.module.state_dict(), os.path.join(args.checkpoint_dir, "simsiam_classifier.pth"))            
