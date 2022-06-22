# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 21:14:06 2022

@author: RitzM
"""

import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import urllib

def object_detection_image():
    st.title('Object Detection for Images')
    st.subheader('''This object detection project takes in an image and''' \
                 ''''outputs the image with bounding boxes created around the objects in the image''')
    file = st.file_uploader('Upload Image', type = ['jpg', 'png', 'jpeg'])
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)
        
        st.image(img1, caption = 'Uploaded Image')
        my_bar = st.progress(0)
        conf_threshold = st.slider('Confidence', 0, 100, 50)
        nms_threshold = st.slider('Threshold', 0, 100, 20)
        whT = 320
        class_names = ['skylight']
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'config_and_weights', 'yolov4-custom.cfg')
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'config_and_weights', 'yolov4-custom_best.weights')
        with open(weights_path, "rb") as response:
            st.write(response.readline())
    
object_detection_image()