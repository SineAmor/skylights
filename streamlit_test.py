import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import os, urllib, cv2

ext_dependencies = {
    'yolov3.weights': {'url': 'https://pjreddie.com/media/files/yolov3.weights',
                       'size': 248007048},
    'yolov3.cfg': {'url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
                   'size': 8342}
}

def download_file(file_path):
    if os.path.exists(file_path):
        if "size" not in ext_dependencies[file_path]:
            return
        elif os.path.getsize(file_path) == ext_dependencies[file_path]['size']:
            return
    
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning('Downloading {}'.format(file_path))
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(ext_dependencies[file_path]['url']) as response:
                length = int(response.info()['Content-Length'])
                counter = 0
                megabytes = 2**20
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    weights_warning.warning('Downloading {}... ({}/{} MB)'.format(file_path, counter/megabytes, length/megabytes))
                    progress_bar.progress(min(counter/length, 1))
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def object_detector_ui():
    confidence_thresh = st.slider('Confidence Threshold', 0, 100, 50)
    overlap_thresh = st.slider('Overlap Threshold', 0, 100, 30)
    return confidence_thresh, overlap_thresh

def yolo_v3(confidence, overlap):
    @st.cache(allow_output_mutation = True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")
    
def main():
    for filename in ext_dependencies.keys():
        download_file(filename)
    st.title('Object Detection for Images')
    st.subheader('''This object detection project takes in an image and''' \
                 ''''outputs the image with bounding boxes created around the objects in the image''')
    file = st.file_uploader('Upload Image', type = ['jpg', 'png', 'jpeg'])
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)
        
        st.image(img1, caption = 'Uploaded Image')
        
if __name__ == "__main__":
    main()