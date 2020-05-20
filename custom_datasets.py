from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy

from utils import *

def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def caltech256_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

def cifar10_transformer():
    return torchvision.transforms.Compose([
        #    transforms.Resize(size=(28, 28)),
           transforms.RandomCrop(32, padding=3),
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #    transforms.Normalize(mean=[0.5, 0.5, 0.5,],std=[0.5, 0.5, 0.5]),
       ])

def cifar100_transformer():
    return torchvision.transforms.Compose([
           transforms.RandomCrop(32, padding=4),
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
           transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
       ])

class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar100_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        # Your transformations here (or set it in CIFAR10)

        return data, target, index

    def __len__(self):
        return len(self.cifar100)

class Caltech256(Dataset):
    def __init__(self, path):
        self.caltech256 =  datasets.ImageFolder(root=path,transform=transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards
            # transforms.Resize((256,256)),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
        ]))

    
    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.caltech256[index]

        return data, target, index

    def __len__(self):
        return len(self.caltech256)

class ImageNet(Dataset):
    def __init__(self, path):
        self.imagenet = datasets.ImageFolder(root=path, transform=imagenet_transformer)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.imagenet[index]

        return data, target, index

    def __len__(self):
        return len(self.imagenet)



class Cityscapes(Dataset):
    def __init__(self, path):
        self.imagenet = datasets.ImageFolder(root=path, transform=imagenet_transformer)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.imagenet[index]

        return data, target, index

    def __len__(self):
        return len(self.imagenet)
