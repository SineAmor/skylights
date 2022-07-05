import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import os, urllib, cv2

ext_dependencies = {
    'yolov4.weights': {'url': 'https://github.com/SineAmor/skylights/raw/main/config_and_weights/yolov4-custom_best.weights',
                       'size': 256093676},
    'yolov4.cfg': {'url': 'https://raw.githubusercontent.com/SineAmor/skylights/main/config_and_weights/yolov4-custom.cfg',
                   'size': 12246}
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

def yolo_v3(img_arr, user_confidence, user_overlap):
    class_names = {0: 'skylights'}
    my_bar = st.progress(0)
    @st.cache(allow_output_mutation = True)
    def load_network(config_path, weights_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        output_layer_names = net.getLayerNames()
        output_layer_names = [output_layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layer_names
    net, output_layer_names = load_network("yolov4.cfg", "yolov4.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def find_objects(outputs):
        hT, wT, cT = img_arr.shape
        bbox, class_ids, confidences = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_ID = np.argmax(scores)
                confidence = scores[class_ID]
                if confidence >= (user_confidence/100):
                    w,h = int(detection[2]*wT), int(detection[3]*hT)
                    x,y = int((detection[0]*wT) - w/2), int((detection[1]*hT) - h/2)
                    bbox.append([x,y,w,h])
                    class_ids.append(class_ID)
                    confidences.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, confidences, user_confidence/100, user_overlap/100)
        results = {}
        for i in indices:
            i = i
            if class_names[class_ids[i]] not in results:
                results[class_names[class_ids[i]]] = 1
            else:
                results[class_names[class_ids[i]]] += 1
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img_arr, (x,y), (x+w, y+h), (255, 0, 0), 2)
            #cv2.putText(img_arr, f'{class_names[class_ids[i]]} {int(confidences[i]*100)}%', (x, y-10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 240), 2)
        st.write(f'{results["skylights"]} skylights detected')
    blob = cv2.dnn.blobFromImage(img_arr, 1/255, (416, 416), [0,0,0], 1, crop = False)
    net.setInput(blob)
    layers_names = net.getLayerNames()
    output_names = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)
    find_objects(outputs)
    
    st.image(img_arr, caption = 'Processed Image')
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    my_bar.progress(100)
    
def main():
    for filename in ext_dependencies.keys():
        download_file(filename)
    confidence_thresh, overlap_thresh = object_detector_ui()
    st.title('Object Detection for Images')
    st.subheader('''This object detection project takes in an image and''' \
                 ''' outputs the image with bounding boxes created around skylights''')
    file = st.file_uploader('Upload Image', type = ['jpg', 'png', 'jpeg'])
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)
        
        st.image(img1, caption = 'Uploaded Image')
        
        yolo_v3(img2, confidence_thresh, overlap_thresh)
        
if __name__ == "__main__":
    main()