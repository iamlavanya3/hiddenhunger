!pip install --upgrade pip
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import requests
from streamlit_lottie import st_lottie
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
st.header("Hidden Hunger in plants")
st.write("To find micronutrient deficiency of banana leaf")

st.markdown("""
    <style>
        .upload-btn-wrapper {
          position: relative;
          overflow: hidden;
          display: inline-block;
        }

        .btn {
          border: 2px solid gray;
          color: gray;
          background-color: white;
          padding: 8px 20px;
          border-radius: 8px;
          font-size: 16px;
          font-weight: bold;
        }

        .upload-btn-wrapper input[type=file] {
          font-size: 100px;
          position: absolute;
          left: 0;
          top: 0;
          opacity: 0;
        }
    </style>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a image file", type="jpg")
model = tf.keras.models.load_model(r"models/resnet152v2.h5")
map_dict = {0: 'Boron deficiency',
            1: 'Healthy',
            2: 'Iron deficiency',
            3: 'Manganese deficiency',
            4: 'Zinc deficiency'}



if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
            
