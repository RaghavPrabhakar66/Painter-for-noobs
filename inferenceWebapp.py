import os 
from PIL import Image
import streamlit as st
import torch

import matplotlib.pyplot as plt

from neuralStyleTransfer import *
from time import time

st.set_option('deprecation.showfileUploaderEncoding', False)
stylePath = {
    'Vangogh - Starry Night': r'serverUpload\testStyle.jpg',
    'Kadinsky - Several Circles': r'Images\Style\Kandinsky-Several Circles.png',
    'Haring - Dance': r'Images\Style\Haring-Dance.png',
    'Picasso - Weeping Woman': r'Images\Style\picasso-weeping woman.png',
    'Vangogh - Whitefield': r'Images\Style\Vangogh-whitefield.png'
}

if __name__ == '__main__':
    st.title('Painter for Noobs')
    content_uploaded_file = st.file_uploader("Choose a image file", type=["png", 'jpg', 'jpeg', 'tiff'], key="1")
    if content_uploaded_file is not None:
        im = Image.open(content_uploaded_file).convert('RGB')
        temp = Image.open(content_uploaded_file).convert('RGB')
        temp = temp.resize((682, 512))
        im.save(r'serverUpload\testContent.jpg', quality=95)
        st.image(temp, caption='Content Image')

    option = st.selectbox( 'Select a Paint Style', ['Vangogh - Starry Night', 'Kadinsky - Several Circles', 'Haring - Dance', 'Picasso - Weeping Woman', 'Vangogh - Whitefield'])

    clicked = st.button('Paint')

    if clicked:
        startTime = time()
        styleImage = neuralStyle(r'serverUpload\testContent.jpg', stylePath['Vangogh - Starry Night'])
        endTime = time()
        im = Image.open('generated1.png').convert('RGB')
        st.image(im, caption='Converted Image')
        st.write('Time Taken : ', (endTime-startTime), ' s')