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
    st.sidebar.markdown('# Model Settings')
    confidence_thresh = st.sidebar.slider('Confidence Threshold', 0, 100, 50)
    overlap_thresh = st.sidebar.slider('Overlap Threshold', 0, 100, 30)
    return confidence_thresh, overlap_thresh

def yolo_v3(image, user_confidence, user_overlap):
    @st.cache(allow_output_mutation = True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")
    
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), [0,0,0], 1, crop = False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)
    
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence >= (user_confidence/100):
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype('int')
                x, y = int(centerX - (width/2)), int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, user_confidence, user_overlap)
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []
    udacity_labels = {
        0: 'pedestrian',
        1: 'biker',
        2: 'car',
        3: 'biker',
        5: 'truck',
        7: 'truck',
        9: 'trafficLight'}
    if len(indices) > 0:
        for i in indices.flatten():
            label = udacity_labels.get(class_IDs[i], None)
            if label is None:
                continue
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            
            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)
            
    boxes = pd.DataFrame({'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax, 'labels':labels})
    return boxes[['xmin', 'ymin', 'xmax', 'ymax', 'labels']]

def draw_image_w_boxes(image, boxes, header, description):
    label_colors = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255]}
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += label_colors[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2
    
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width = True, caption = 'Processed Image')
        
def main():
    for filename in ext_dependencies.keys():
        download_file(filename)
    confidence_thresh, overlap_thresh = object_detector_ui()
    st.title('Object Detection for Images')
    st.subheader('''This object detection project takes in an image and''' \
                 ''''outputs the image with bounding boxes created around the objects in the image''')
    file = st.file_uploader('Upload Image', type = ['jpg', 'png', 'jpeg'])
    if file != None:
        img1 = Image.open(file)
        st.image(img1, caption = 'Uploaded Image')
        yolo_boxes = yolo_v3(img1, confidence_thresh, overlap_thresh)
        draw_image_w_boxes(img1, yolo_boxes, 'Real-time Computer Vision')
        
if __name__ == "__main__":
    main()