import os 
from PIL import Image
import streamlit as st
import torch

import matplotlib.pyplot as plt

from neuralStyleTransfer import *
import time

st.set_option('deprecation.showfileUploaderEncoding', False)
stylePath = {
    'Vangogh - Starry Night'    : r'Images\Style\vangogh.png',
    'Kadinsky - Several Circles': r'Images\Style\Kandinsky-Several Circles.png',
    'Haring - Dance'            : r'Images\Style\Haring-Dance.png',
    'Picasso - Weeping Woman'   : r'Images\Style\picasso-weeping woman.png',
    'Vangogh - Whitefield'      : r'Images\Style\Vangogh-whitefield.png'
}

if __name__ == '__main__':
    st.title('Painter for Noobs')

    st.sidebar.header('Configurations')

    option = st.sidebar.selectbox( 'Select a Paint Style', ['Vangogh - Starry Night', 'Kadinsky - Several Circles', 'Haring - Dance', 'Picasso - Weeping Woman', 'Vangogh - Whitefield'])
    showStyleImage = st.sidebar.checkbox("Want to see the image", key='1')

#    epochs  = st.sidebar.slider('Epochs', 0, 1000, 300)
    stylize = st.sidebar.slider('Stylize Weight')
    showStatus = st.sidebar.button('Submit')

    if showStyleImage:
        style = Image.open(stylePath[option]).convert('RGB')
        st.sidebar.image(style, caption='Style Image', use_column_width=True)

    content_uploaded_file = st.file_uploader("Choose a image file", type=["png", 'jpg', 'jpeg', 'tiff'], key="1")
    showContentImage = st.checkbox("Want to see the image", key='2')

    if content_uploaded_file is not None:
        im = Image.open(content_uploaded_file).convert('RGB')
        im.save(r'serverUpload\testContent.jpg', quality=95)
        if showContentImage:
            st.image(im, caption='Content Image', use_column_width=True)

    clicked = st.button('Paint')

    if clicked:
        startTime = time.time()
        myProgressBar = st.progress(0)
        device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = models.vgg19(pretrained=True).features

        for param in model.parameters():
            param.requires_grad = False

        model.to('cuda:0')

        contentImg, styleImg = readImgs(r'serverUpload\testContent.jpg', os.path.abspath(stylePath[option]))

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

            print("epoch : {} Total Loss : {}".format(i, totalLoss))

            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            if i%3==0:
                myProgressBar.progress(int(i/3))

        trainStyle = convertFromTensor(target)
        utils.save_image(target, 'generated1.png')

        endTime = time.time()
        im = Image.open('generated1.png').convert('RGB')
        st.image(im, caption='Converted Image')
        st.write('Time Taken : ', (endTime-startTime), ' s')