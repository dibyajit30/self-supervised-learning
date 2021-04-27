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
from lr_scheduler import LR_Scheduler

train_transform = transforms.Compose([
    transforms.ToTensor(),
])


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
args = parser.parse_args()

unlabeledset = CustomDataset(root='/home/jupyter/dataset', split="unlabeled", transform=train_transform)
unlabeledloader = torch.utils.data.DataLoader(unlabeledset, batch_size=256, shuffle=True, num_workers=2)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
print(device)
#encoder_checkpoint_path = os.path.join(args.checkpoint_path, "simsiam_encoder_1.pth")
#encoder_checkpoint_path = "/home/jupyter/simsiam_encoder_1.pth"
#checkpoint=torch.load(encoder_checkpoint_path)
model=get_encoder()
#model.load_state_dict(checkpoint)
model=torch.nn.DataParallel(model)
model=model.to(device)

lr = 0.05 * 256 / 256
predictor_prefix = ('module.predictor', 'predictor')
parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]
optimizer = torch.optim.SGD(parameters,lr=lr,momentum=0.9,weight_decay=5e-4)
lr_scheduler = LR_Scheduler(
        optimizer,
        1, 0, 
        100, 0.05*256/256, 0, 
        len(unlabeledloader),
        constant_predictor_lr=True 
)
print("Started contrastive training")
cur = time.time()
for epoch in range(100):
    model.train()
    running_loss = 0.0
#     optimizer.zero_grad()
#     batch=1
    for i, data in enumerate(unlabeledloader):
        inputs,_ = data
        inputs1,inputs2 = generate_pairs_simsiam(inputs)
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        loss = model(inputs1,inputs2)
#         loss=loss/4
        loss=loss.mean()
        optimizer.zero_grad()
        loss.backward()
#         if batch%4==0:
        optimizer.step()
        lr_scheduler.step()
#             optimizer.zero_grad()
        running_loss += loss.item()
        if (i) % 100 == 99:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' % (epoch + 1, i, running_loss / 10))
            running_loss = 0.0
#         batch+=1
    filename="simsiam_encoder_"+str(epoch)+".pth"
    torch.save(model.module.state_dict(), '/home/jupyter/self-supervised-learning/checkpoint96/'+filename)

print("Training time {}".format(time.time()-cur))
os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, "simsiam_encoder_96.pth"))