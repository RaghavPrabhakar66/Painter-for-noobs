import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import cuda
from torch.cuda import is_available
from torchvision import *
import pickle

def modelActivation(x, model):
    layers = {
        '0'  : 'conv1_1',
        '5'  : 'conv2_1',
        '10' : 'conv3_1',
        '19' : 'conv4_1',
        '21' : 'conv4_2',
        '28' : 'conv5_1',
    }
    features = {}
    x = x.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    
    return features

def readImgs(contentPath, stylePath):
    contentPath = os.path.abspath(contentPath)
    stylePath   = os.path.abspath(stylePath)

    contentImg = Image.open(contentPath)
    styleImg   = Image.open(stylePath)

    return contentImg, styleImg

def convertToTensor(imageTensor):
    transform = transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    imageTensor = transform(imageTensor).to(device)

    return imageTensor

def convertFromTensor(imageTensor):
    x = imageTensor.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)
    x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return x


def gramMatrix(imgFeatures):
    _, d, h, w  = imgFeatures.size()
    imgFeatures = imgFeatures.view(d, h*w)
    Matrix = torch.mm(imgFeatures, imgFeatures.t())

    return Matrix

if __name__ == '__main__':

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device = ",device)

    model = models.vgg19(pretrained=True).features

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)

    print(model)

    contentImg, styleImg = readImgs(r'Images\Content\rose.jpg', r'Images\Style\batman.jpg')

    contentImg = convertToTensor(contentImg)
    styleImg   = convertToTensor(styleImg)

    contentFeatures = modelActivation(contentImg, model)
    styleFeatures   = modelActivation(styleImg, model)

    styleWeight  = {"conv1_1" : 1.0, 
                    "conv2_1" : 0.8,
                    "conv3_1" : 0.4,
                    "conv4_1" : 0.2,
                    "conv5_1" : 0.1}

    styleGram = {layer:gramMatrix(styleFeatures[layer]) for layer in styleFeatures}

    content_wt = 100
    style_wt = 1e8

    print_after = 500
    epochs = 4000

    target = contentImg.clone().requires_grad_(True).to(device)

    optimizer = torch.optim.Adam([target],lr=0.007)

    for i in range(epochs):
        targetFeatures  = modelActivation(target, model)
        contentLoss = torch.mean((contentFeatures['conv4_2'] - targetFeatures['conv4_2'])**2)

        styleLoss = 0
        for layer in styleWeight:
            styleGramMatrix  = styleGram[layer]
            targetGramMatrix = targetFeatures[layer]
            _, c, h, w       = targetGramMatrix.shape
            targetGramMatrix = gramMatrix(targetGramMatrix) 
            styleLoss       += torch.mean(styleWeight[layer]*torch.mean((targetGramMatrix-styleGramMatrix)**2)) / c*h*w
        
        totalLoss = styleLoss + contentLoss

        print("epoch : {} Total Loss : {}".format(i, totalLoss))

        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

        #if i%print_after == 0:
        #    trainStyle = convertFromTensor(target)
        
        #    plt.imshow(trainStyle, label = "Style")
        #    plt.show()

    torch.save(model.state_dict(), os.path.abspath(r'pretrained-models\model.pt'))
    trainStyle = convertFromTensor(target)
    plt.imshow(trainStyle, label = "Style")
    plt.show()
