import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def prepare_data(path_data, batch_size):
    cifar_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])  # rescale image to from -1 to 1
    traids = torchvision.datasets.CIFAR10(root=path_data, transform=cifar_transform,
                                          train=True, download=True)
    testds = torchvision.datasets.CIFAR10(root=path_data, transform=cifar_transform,
                                          train=False, download=True)
    trainloader = torch.utils.data.DataLoader(trainds, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(trainds, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    return trainds, trainloader, testds, testloader

def plot_image(images, labels):
    img_grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    np_img = img_grid.numpy().transpose(1,2,0) # pytorch has the order, c,w,h
    # to be able to view an image, we need to chage the order and 
    # put it in width, height, color order 
    plt.imshow(np_img)
