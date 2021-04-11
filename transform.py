import torchvision
import torch

def get_color_distortion(s=1):
    color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter =  torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray =  torchvision.transforms.RandomGrayscale(p=0.2)
    color_distort =  torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def generate_pairs(images):
    images1 , images2 = [], []
    for img in images:
        img = torchvision.transforms.ToPILImage()(img)
        img1 = get_color_distortion()(img)
        img2 = get_color_distortion()(img)
        images1.append(torchvision.transforms.ToTensor()(img1))
        images2.append(torchvision.transforms.ToTensor()(img2))
    images1 , images2 = torch.stack(images1), torch.stack(images2)
    return images1, images2