import streamlit as st
from PIL import Image
import torchutils as tu
import torch
import torchvision
import torch.nn as nn
import numpy as np
import requests
import base64
import time

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
# from torchvision import io
from pages.firstmodel.models import model_one
from pages.firstmodel.preprocessing import preprocess
from io import BytesIO


@st.cache_resource()



#model = model_two
def load_model():
  model = model_one
  model.load_state_dict(torch.load('pages/firstmodel/savemodelone.pt'))
  return model
#   return model


model = load_model()

model.eval()


st.title('Напоследок посмотреть на природу')

uploaded_images = st.file_uploader("Загрузите изображения", type=["jpg", "png"], accept_multiple_files=True)
# image = st.file_uploader('Загрузи файл')
image_url = st.text_input("Введите URL изображения для загрузки")


class_labels = {
    0: "buildings",
    1: "forest",
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street"
}

def predict(img):
  start_time = time.time()
  img = preprocess(img)
  prediction = torch.argmax(model(img.unsqueeze(0)),dim=1).item()
  pred = class_labels[prediction]
  end_time = time.time()

  inference_time = end_time - start_time
  st.write(f"Время выполнения предсказания: {inference_time:.2f} секунд")
  return pred


if uploaded_images:
    for image in uploaded_images:
        image = Image.open(image)
        prediction = predict(image)
        st.image(image)
        st.write(f"Предсказание: {prediction}")


if image_url:
    try:
        if image_url.startswith("data:image"):
            # Handle data URI
            image_data = image_url.split(',')[1]
            image_binary = base64.b64decode(image_data)
            image_url = Image.open(BytesIO(image_binary))
        else:
            # Download the image from the URL
            response = requests.get(image_url)
            if response.status_code == 200:
                image_data = BytesIO(response.content)
                image_url = Image.open(image_data)
            else:
                st.error("Error loading the image from the URL")
                st.stop()

        prediction = predict(image_url)
        st.image(image_url, caption='Image from URL', use_column_width=True)
        st.write(f'Prediction: {prediction}')
    except Exception as e:
        st.error(f"Error loading the image: {str(e)}")


# def get_prediction(path: str) -> str:
#   model.eval()
#   prediction = torch.sigmoid(model(img.unsqueeze(0).to(device))).round().item()
  
#   if prediction == 0:
#     prediction = 'Cat'
#   elif prediction == 1:
#     prediction = 'Dog'

#   return prediction

# image = st.file_uploader('Загрузи файл')

# if image:
#   image = Image.open(image)
#   prediction = get_prediction(image)
#   st.image(image)
#   st.write(prediction)