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

def object_detection_image():
    st.title('Object Detection for Images')
    st.subheader('''This object detection project takes in an image and''' \
                 ''''outputs the image with bounding boxes created around the objects in the image''')