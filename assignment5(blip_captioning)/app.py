# frontend_streamlit.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="🖼️ BLIP Image Captioning", layout="centered")
st.title("Image Captioning with BLIP")
st.caption("Upload an image or enter an image URL to generate a caption using a powerful multimodal model (BLIP).")

backend_url = "http://localhost:8000"

option = st.radio("Choose input method:", ["Upload Image", "Image URL"])

if option == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        st.image(file, caption="Uploaded Image", width=400)
        if st.button("Generate Caption"):
            with st.spinner("Captioning..."):
                response = requests.post(f"{backend_url}/caption/upload", files={"file": file})
                result = response.json()
                st.write(result.get("caption", "No caption returned."))

elif option == "Image URL":
    url = st.text_input("Enter image URL:")
    if url:
        try:
            img = Image.open(requests.get(url, stream=True).raw)
            st.image(img, caption="Image from URL", width=400)
            if st.button("Generate Caption"):
                with st.spinner("Captioning..."):
                    response = requests.post(f"{backend_url}/caption/url", json={"url": url})
                    result = response.json()
                    st.write(result.get("caption", "No caption returned."))
        except Exception as e:
            st.error(f"Failed to load image: {e}")
