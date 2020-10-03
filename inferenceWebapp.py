import os 
from PIL import Image
import streamlit as st
import torch

import matplotlib.pyplot as plt

from neuralStyleTransfer import *
from time import time

st.set_option('deprecation.showfileUploaderEncoding', False)

if __name__ == '__main__':
    st.title('Painter for Noobs')
    content_uploaded_file = st.file_uploader("Choose a image file", type=["png", 'jpg', 'jpeg', 'tiff'], key="1")
    if content_uploaded_file is not None:
        im = Image.open(content_uploaded_file).convert('RGB')
        im.save(r'serverUpload\testContent.jpg', quality=95)
        st.image(im, caption='Content Image')

    style_uploaded_file = st.file_uploader("Choose a image file", type=["png", 'jpg', 'jpeg', 'tiff'], key="2")
    if style_uploaded_file is not None:
        im = Image.open(style_uploaded_file).convert('RGB')
        im.save(r'serverUpload\testStyle.jpg', quality=95)
        st.image(im, caption='Style Image')

    #startTime = time()
    #styleImage = neuralStyle(r'serverUpload\testContent.jpg', r'serverUpload\testStyle.jpg')
    #endTime = time()        
    #print('Time Taken : ', (endTime-startTime))
    #st.image(styleImage, caption='Converted Image')