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
from transform import generate_pairs_simsiam, Cutout
import time
train_transform = transforms.Compose([
            T.RandomResizedCrop((96,96), scale=(0.2, 1.0)),
            #T.Resize((128,128)),
            #T.CenterCrop((96,96)),
            T.RandomHorizontalFlip(),
            #T.RandomVerticalFlip(p=0.2),
            T.RandomApply([T.RandomRotation((-45,45))],p=0.1),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=96//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor()
])

val_transform = transforms.Compose([
    #T.Resize((224,224)),
    transforms.ToTensor()
])
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()
#encoder_checkpoint_path = os.path.join(args.checkpoint_path, "simsiam_encoder.pth")
encoder_checkpoint_path = "/home/jupyter/self-supervised-learning/checkpoint96/simsiam_encoder_19.pth"
trainset = CustomDataset(root='/home/jupyter/dataset', split="train", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=1)
checkpoint=torch.load(encoder_checkpoint_path)
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model=get_encoder()
model.load_state_dict(checkpoint)
#model=torch.nn.DataParallel(model)
model=model.to(device)

classifier = get_classifier()
#classifier=torch.nn.DataParallel(classifier)
classifier=classifier.to(device)
valset1 = CustomDataset(root='/home/jupyter/dataset', split="train", transform=val_transform)
valloader1 = torch.utils.data.DataLoader(valset1, batch_size=256, shuffle=False, num_workers=1)

valset = CustomDataset(root='/home/jupyter/dataset', split="val", transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=1)
def train(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valloader1:
            images, labels = data

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            #outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct/total)

def val(model):
    #print("Started validating")
    correct = 0
    total = 0
    model.eval()
    classes={}
    for i in range(800):
        classes[i]=0
    with torch.no_grad():
        for data in valloader:
            images, labels = data

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            #outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
#             for i in range(len(predicted)):
#                 if predicted[i].item()!=labels[i].item():
#                     classes[labels[i].item()]+=1
    return (correct/total) 

batch=1
print("Started Supervised training")
model=model.backbone
model.fc=classifier.classifier
model=model.to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,nesterov='True',momentum=0.9)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.1)
for epoch in range(100):
        model.train()
        running_loss = 0.0
        for idx, (images, labels) in enumerate(trainloader):
            preds=model(images.to(device))
            loss = F.cross_entropy(preds, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            if (idx) % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.6f' % (epoch + 1, idx+1, running_loss / 10))
                running_loss = 0.0
        train_acc=train(model)
        val_acc=val(model)
        #scheduler.step()
        print("Training acc {} Val Acc {}".format(train_acc,val_acc))
        fname="checkpointfinal96/simsiam_fine_tuned_96_"+str(epoch)+".pth"
        torch.save(model.state_dict(), fname)            



#print("Started Training acc")

# print(classes)
# import pickle
# pickle.dump(classes,open('labels.pickle','wb'))