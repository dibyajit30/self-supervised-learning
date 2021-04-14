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
from submission_simsiam import get_encoder, D
from transform import generate_pairs_simsiam
import time
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

unlabeledset = CustomDataset(root='/dataset', split="unlabeled", transform=train_transform)
unlabeledloader = torch.utils.data.DataLoader(unlabeledset, batch_size=256, shuffle=True, num_workers=2)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model=get_encoder()
model=torch.nn.DataParallel(model)
model=model.to(device)

lr = 0.05 * 256 / 256
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)

print("Started contrastive training")
batch=1
cur = time.time()
for epoch in range(1):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(unlabeledloader):
        inputs,_ = data
        inputs1,inputs2 = generate_pairs_simsiam(inputs)
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        loss = model(inputs1,inputs2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i) % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i, running_loss / 10))
            running_loss = 0.0
        batch+=1
print("Training time {}".format(time.time()-cur))
os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, "simsiam_encoder.pth"))