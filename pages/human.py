import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
st.header("Hidden Hunger")
st.write("To find micronutrient deficiency in human using the images of nails and eyes")
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
model = tf.keras.models.load_model(r"D:\project\resnet152v2nail.h5")
camera=st.button("Capture")
if camera:
    # Function to process each frame of the video
    def process_frame(frame):
    # Convert the frame to grayscale
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to the grayscale image
      _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
      return thresh

# Create a VideoCapture object to capture video from the camera
    cap = cv2.VideoCapture(0)

# Set the dimensions of the video capture window
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define a function to capture video and display the results in the Streamlit app
    def capture_video():
        while True:
        # Read a frame from the camera
          ret, frame = cap.read()
        
        # Process the frame
          processed_frame = process_frame(frame)
        
        # Display the original and processed frames in the Streamlit app
          st.image(np.hstack((frame, processed_frame)), width=640)
        
        # Check if the user has pressed the "Stop" button
          if st.button('Stop'):
                break

# Call the function to capture video and display the results in the Streamlit app
    capture_video()

# Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()
map_dict = {0: 'Iodine deficiency',
            1: 'Vitamin B12 deficiency',
            2: 'Vitamin D deficiency',
            3: 'Zinc deficiency',
            4: 'Healthy',
            5: 'Iron deficiency'}

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
        
