import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from custom_loss import nt_xent

from dataloader import CustomDataset
from submission import get_model, get_projection_head, LinearNet
from transform import generate_pairs
from custom_optimizer import LARS_simclr

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

continue_training = False

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((96,96), scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=96//20*2+1, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
])

trainset = CustomDataset(root='/dataset', split="train", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

unlabeledset = CustomDataset(root='/dataset', split="unlabeled", transform=train_transform)
unlabeledloader = torch.utils.data.DataLoader(unlabeledset, batch_size=256, shuffle=True, num_workers=2)

if torch.cuda.is_available():
  device = torch.device("cuda")
  
else:
  device = torch.device("cpu")
  print('GPU not found, training will be slow...')

#net = get_classifier_model()
combined_net = get_model()
net = combined_net.encoder
net = torch.nn.DataParallel(net)
#net = net.to(device)

net.module.model.fc = get_projection_head(net.module.model)
if continue_training:
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "simclr_encoder.pth"))
    net.load_state_dict(checkpoint)
net = net.to(device)

#optimizer = torch.optim.SGD(net.parameters(), lr=0.3)
optimizer = LARS_simclr(net.named_modules(), lr=0.3)

print('Start Training')

print("Contrastive training")
net.train()

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(unlabeledloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data
        #inputs = inputs.to(device)
        inputs1, inputs2 = generate_pairs(inputs)
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)

        outputs1 = net(inputs1)
        outputs2 = net(inputs2)
        loss = nt_xent(outputs1, outputs2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i+1)))
    # Save checkpoint
    torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, "simclr_encoder.pth"))

print("Supervised training")

# Removing the projection head
net.module.model.fc = net.module.model.fc[0]
# Save model for backup
torch.save(combined_net.state_dict(), os.path.join(args.checkpoint_dir, "simclr.pth"))
#net.eval()
#classifier = LinearNet(encoder_features=net.module.fc[-1].out_features)
'''classifier = combined_net.classifier
classifier = torch.nn.DataParallel(classifier)
classifier = classifier.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(combined_net.parameters(), lr=0.1)
classifier.train()


for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        features = net(inputs)
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0'''

print('Finished Training')

'''os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(combined_net.state_dict(), os.path.join(args.checkpoint_dir, "simclr.pth"))'''

print(f"Saved encoder checkpoint to {os.path.join(args.checkpoint_dir, 'simclr.pth')}")
