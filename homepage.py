import streamlit as st
import requests
import PIL
from PIL import Image



st.set_page_config(
    page_title="Hidden hunger",
    page_icon="👁🌿"
)

st.title("Hidden Hunger Detection")
st.subheader("This website is used to find the micronutrient deficiency of human and plants.")


st.write("Hidden hunger in human is found by using the images of nails and eyes.The micronutrient deficiency which can deteced for human are iron,iodine,vitamin b12,vitamin d,zinc and healthy")
st.write("Hidden hunger in plant is found by using the images of banana leaf.The micronutrient deficiency which can deteced for human are iron,zinc,manganese,boron and healthy")

st.sidebar.success("Select a page above.")
        
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")


# Load the image file
image = Image.open('micro.jpg')
# Display the image in the Streamlit app
st.image(image, use_column_width=True)

