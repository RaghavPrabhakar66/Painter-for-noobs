import os

import numpy as np
import torch
from PIL import Image
from torchvision import *
import streamlit as st

from inferenceWebapp import *

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

    contentImg = Image.open(contentPath).convert('RGB')
    styleImg   = Image.open(stylePath).convert('RGB')

    return contentImg, styleImg

def convertToTensor(image):
    transform = transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
    image = transform(image).to('cuda:0')

    return image

def convertFromTensor(imageTensor):
    x = imageTensor.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)
#    x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return x

def gramMatrix(imgFeatures):
    _, d, h, w  = imgFeatures.size()
    imgFeatures = imgFeatures.view(d, h*w)
    Matrix = torch.mm(imgFeatures, imgFeatures.t())

    return Matrix

def neuralStyle(contentPath, stylePath):
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.vgg19(pretrained=True).features

    for param in model.parameters():
        param.requires_grad = False

    model.to('cuda:0')

    contentImg, styleImg = readImgs(contentPath, stylePath)

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

    epochs = 300

    target = contentImg.clone().requires_grad_(True).to()

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

        #print("epoch : {} Total Loss : {}".format(i, totalLoss))

        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

    trainStyle = convertFromTensor(target)
    utils.save_image(target, 'generated1.png')
    trainStyle = convertFromTensor(target)

    return trainStyle