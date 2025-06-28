import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from io import BytesIO

app = FastAPI(title = "Blip for Image Captioning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_url = 'https://www.purina.in/sites/default/files/2023-05/feast.png'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ImageURLRequest(BaseModel):
    url : str
    
#Caption Generation Logic
def generate_caption(image: Image.Image) -> str:
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

#Upload image endpoint
@app.post("/caption/upload")
async def caption_from_upload(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        caption = generate_caption(image)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}

# Image URL endpoint
@app.post("/caption/url")
async def caption_from_url(data: ImageURLRequest):
    try:
        response = requests.get(data.url, stream=True)
        image = Image.open(response.raw)
        caption = generate_caption(image)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}