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
from pages.secondmodel.models import model_two
from pages.secondmodel.preprocessing import preprocess
from io import BytesIO


@st.cache_resource()



#model = model_two
def load_model():
  model = model_two
  model.load_state_dict(torch.load('pages/secondmodel/savemodeltwo.pt'))
  return model


model = load_model()

model.eval()


st.title('Давай посмотрим будешь жить или нет')


uploaded_images = st.file_uploader("Загрузите изображения", type=["jpg", "png"], accept_multiple_files=True)
# image = st.file_uploader('Загрузи файл')
image_url = st.text_input("Введите URL изображения для загрузки")

# Проверить, если пользователь ввел URL изображения





def predict(img):
  start_time = time.time()
  img = preprocess(img)
  prediction = torch.sigmoid(model(img.unsqueeze(0))).round().item()
  class_to_label = {0: 'benign', 1: 'malignant'}
  pred = class_to_label[prediction]
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