import os
import random
from dotenv import load_dotenv
from sqlalchemy.testing.plugin.plugin_base import logging
import time
load_dotenv() # Load all the environment variables from env file

import streamlit as st
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GENAI_API_KEY"))
# function to load Gemini pro vision
# model=genai.GenerativeModel('gemini-pro-vision')
model=genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input,image[0],prompt])
    return response.text

def input_image_config(img):
    if img is not None:
        # Read the files into bytes
        bytes_data = img.getvalue()

        image_parts=[
            {
                "mime_type": img.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("Error in file upload...")

# Initialize Streamlit app

st.set_page_config(page_title="Multilanguage Invoice Extractor", layout="wide")
st.header("Multilanguage Invoice Extractor")
input = st.text_input(label="Input prompt: ", key="input")
uploaded_file = st.file_uploader("Upload your invoice here...", type=["jpg","png","jpeg"])

try:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)
    else:
        st.warning("Remember to upload your invoice!!!")
except Exception as e:
    # logging.error(e)
    st.text("An unknown error has occurred, Please try again...")
    st.error(e)


emojis = ["😀", "🚀", "🔥", "✨", "🎉", "🏆", "💡", "🍕", "🎵", "🌟", "🔥", "💡","🧐","😺","🤞","🎩","⚡️","🎯","🚦","🔦","💯","🔍"]
# Function to select 3 random emojis
def select_random_emojis():
    return random.sample(emojis, 3)
# Add CSS animations for rotating/blinking emojis
def add_emoji_animations():
    st.markdown(
        """
        <style>
        .blinking-emoji {
            animation: blinker 1s linear infinite;
        }
        @keyframes blinker {
            50% { opacity: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# Display emojis with animations
def display_loading_emojis(emojis, placeholder):
    placeholder.markdown(
        f"""
        <div style="text-align: center; font-size: 48px;">
            <span class="blinking-emoji">{emojis[0]}</span>
            <span class="blinking-emoji">{emojis[1]}</span>
            <span class="blinking-emoji">{emojis[2]}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


submit=st.button("Analyze your invoice")

base_prompt="""
    You are an expert in understanding invoices. We will upload an image as a invoice and you have to answer any question based on uploaded invoice image.
"""


if submit:
    # Select 3 random emojis
    emojis = select_random_emojis()
    # Add CSS for animations
    add_emoji_animations()
    # Create a placeholder for the loading animation
    loading_placeholder = st.empty()

    display_loading_emojis(emojis, loading_placeholder)

    image_data = input_image_config(uploaded_file)
    response = get_gemini_response(base_prompt, image_data, input)
    time.sleep(3)
    loading_placeholder.empty()
    st.write(response)

