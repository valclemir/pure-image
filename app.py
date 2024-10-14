import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import torch
import os

# Definir dispositivo para o modelo YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar modelo YOLO
DIR = os.getcwd() + "/pure-image/best.pt"  # Defina o caminho do modelo YOLO
model = YOLO(DIR).to(device)

def apply_inpaint_on_detections(img, results, inpaint_radius=7):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    inpainted_img = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted_img

def adjust_brightness(img, bright=100):
    bright = bright - 100
    img_brightness = cv2.convertScaleAbs(img, alpha=1, beta=bright)
    return img_brightness

def adjust_contrast(img, contrast=105):
    contrast = contrast / 100
    img_contrast = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
    return img_contrast

def adjust_sharpness(img, intensidade=0.1):
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9 + intensidade, -1],
                               [-1, -1, -1]])
    sharpened_image = cv2.filter2D(img, -1, kernel_sharpen)
    return sharpened_image

def process_image(img, inpaint_radius, brightness, contrast, sharpness_intensity):
    # Fazer predições com o YOLO
    results = model.predict(img, imgsz=3616, conf=0.01)

    # Aplicar apenas o inpainting nas detecções
    img_with_inpaint = apply_inpaint_on_detections(img.copy(), results, inpaint_radius)

    # Ajustar brilho, contraste e nitidez
    img_brightness = adjust_brightness(img_with_inpaint, bright=brightness)
    img_contrast = adjust_contrast(img_brightness, contrast=contrast)
    img_sharpened = adjust_sharpness(img_contrast, intensidade=sharpness_intensity)

    return img_sharpened

# Título do app
st.title("Ajuste de Imagem: Contraste, Brilho, Nitidez e Inpainting.")

# Carregar imagem
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Carregar a imagem no formato PIL
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Sliders para os parâmetros
    #inpaint_radius = st.slider("Raio para Inpainting", 1, 15, 7)
    inpaint_radius = 0.5
    brightness = st.slider("Ajuste de Brilho", 0, 200, 100)
    contrast = st.slider("Ajuste de Contraste", 50, 200, 105)
    sharpness_intensity = st.slider("Intensidade de Nitidez", 0.1, 2.0, 0.5)

    # Processar a imagem com as detecções YOLO e ajustes
    processed_image = process_image(img, inpaint_radius, brightness, contrast, sharpness_intensity)

    # Dividir a tela em duas colunas para exibir imagens lado a lado
    col1, col2 = st.columns(2)

    # Exibir imagem original na primeira coluna
    with col1:
        st.image(image, caption='Imagem Original', use_column_width=True)

    # Exibir imagem processada na segunda coluna
    with col2:
        st.image(processed_image, caption='Imagem Processada', use_column_width=True)
