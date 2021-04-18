import torch
from submission_simsiam import get_encoder, D, get_classifier
from torch import nn
from dataloader import CustomDataset
from torchvision import transforms
import torchvision.transforms as T
import pickle
import numpy as np

checkpoint=torch.load('/mnt/e/Downloads/simsiam_fine_tuned.pth',map_location=torch.device('cpu'))
model=get_encoder()
model=model.backbone
classifier=get_classifier()
model.fc=classifier.classifier
model.load_state_dict(checkpoint)
train_transform = transforms.Compose([
            T.ToTensor()
])
trainset = CustomDataset(root='/mnt/e/Downloads/student_dataset/dataset/', split="train", transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=1)
softmax=[]
for i,(data,_) in enumerate(trainloader):
    softmax.append(model(data).data)
x=torch.stack(softmax)
x=x.reshape(x.shape[0]*x.shape[1],x.shape[2])

#classes=pickle.load(open('labels.pickle','rb'))
classes=np.arange(0,100)
elems=128
indices_list=set()
for class_id in classes:
    arr=x[:,class_id]
    arr,indices=torch.sort(arr,descending=True)
    indices=indices.numpy()
    i=0
    j=0
    while i < elems and j<indices.shape[0]:
        if indices[j] not in indices_list:
            indices_list.add(indices[j])
            i+=1
        j+=1
print(indices_list)