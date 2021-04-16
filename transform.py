import torchvision.transforms as T
import torch

def get_color_distortion(s=1):
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter =  T.RandomApply([color_jitter], p=0.8)
    rnd_gray =  T.RandomGrayscale(p=0.2)
    color_distort =  T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def get_other_transformation():
    transform = T.Compose([
            T.RandomResizedCrop((96,96), scale=(0.2, 1.0)),
            #T.RandomHorizontalFlip(),
            #T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            #T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=96//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor()
        ])
    return transform

def generate_pairs(images):
    images1 , images2 = [], []
    for img in images:
        img = T.ToPILImage()(img)
        img1 = get_color_distortion()(img)
        img2 = get_color_distortion()(img)
        img1 = get_other_transformation()(img1)
        img2 = get_other_transformation()(img2)
        images1.append(img1)
        images2.append(img2)
    images1 , images2 = torch.stack(images1), torch.stack(images2)
    return images1, images2