import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from dataloader import CustomDataset
from submission import get_model, eval_transform, team_id, team_name, email_address, LinearNet

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=str)
args = parser.parse_args()

evalset = CustomDataset(root='/dataset', split="val", transform=eval_transform)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=256, shuffle=False, num_workers=2)

encoder_checkpoint_path = os.path.join(args.checkpoint_path, "simclr_encoder.pth")
classifier_checkpoint_path = os.path.join(args.checkpoint_path, "simclr_classifier.pth")

encoder = get_model()
encoder.fc = encoder.fc[:-2]
checkpoint = torch.load(encoder_checkpoint_path)
encoder.load_state_dict(checkpoint)
encoder = encoder.cuda()
encoder.eval()

classifier = LinearNet(encoder_features=encoder.fc[-1].out_features)
checkpoint = torch.load(classifier_checkpoint_path)
classifier.load_state_dict(checkpoint)
classifier = classifier.cuda()
classifier.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in evalloader:
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        features = encoder(images)
        outputs = classifier(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f"Team {team_id}: {team_name} Accuracy: {(100 * correct / total):.2f}%")
print(f"Team {team_id}: {team_name} Email: {email_address}")
