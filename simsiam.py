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
from submission_simsiam import get_model, D
from torch.optim import lr_scheduler
from transform import generate_pairs_simsiam
import time
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

unlabeledset = CustomDataset(root='/dataset', split="unlabeled", transform=train_transform)
unlabeledloader = torch.utils.data.DataLoader(unlabeledset, batch_size=512, shuffle=True, num_workers=2)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model=get_model()
model=torch.nn.DataParallel(model)
model=model.to(device)

lr = 0.05 * 512 / 256
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

print("Started contrastive training")
batch=1
for epoch in range(1):
    cur = time.time()
    model.train()
    running_loss = 0.0
    for i, data in enumerate(unlabeledloader):
        inputs,_ = data
        inputs1,inputs2 = generate_pairs_simsiam(inputs)
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        loss = model(inputs1,inputs2)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        running_loss += loss.item()
        batch+=1
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
    print("Training time {}".format(time.time()-cur))
os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, "simsiam_encoder.pth"))