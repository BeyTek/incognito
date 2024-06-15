import os
import cv2
import numpy as np
import uuid
import torch
import torchvision
import random
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import time
import threading

st.set_page_config(
    page_title="Incognito BeyTek",
    page_icon="🤐",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.to(device)
model.eval()

epsilon = 0.02
image = None
perturbed_image = None

st.title("Adversarial Attack By BeyTek")

uploaded_image = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])

# Add a dropdown to select the attack method
attack_method = st.selectbox("Choose Attack Method", ["FGSM", "PGD", "DeepFool"])

epsilon_slider = st.slider("Attack Intensity", min_value=2, max_value=40, value=6)

def fgsm_attack(image_tensor, epsilon):
    output = model(image_tensor)
    criterion = torch.nn.CrossEntropyLoss()
    target = torch.tensor([random.randint(0, output.shape[1] - 1)]).to(device)
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = image_tensor.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(image_tensor, epsilon, alpha=0.01, num_iter=40):
    perturbed_image = image_tensor.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        criterion = torch.nn.CrossEntropyLoss()
        target = torch.tensor([random.randint(0, output.shape[1] - 1)]).to(device)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
        eta = torch.clamp(perturbed_image - image_tensor, -epsilon, epsilon)
        perturbed_image = torch.clamp(image_tensor + eta, 0, 1).detach_()
    return perturbed_image

if st.button("Attack Image"):
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = image.convert('RGB')
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        
        epsilon = epsilon_slider / 100.0

        if attack_method == "FGSM":
            perturbed_image = fgsm_attack(image_tensor, epsilon)
        elif attack_method == "PGD":
            perturbed_image = pgd_attack(image_tensor, epsilon)
        elif attack_method == "DeepFool":
            st.write("DeepFool not implemented yet. Please choose another method.")
            perturbed_image = None
        
        if perturbed_image is not None:
            perturbed_image = torchvision.transforms.ToPILImage()(perturbed_image.squeeze(0).cpu())

            perturbed_image_np = np.array(perturbed_image)
            random_uuid = str(uuid.uuid1())[:6]
            output_filename = f"{random_uuid}-Atk.jpg"
            
            watermark = Image.new("RGBA", perturbed_image.size)
            draw = ImageDraw.Draw(watermark)
            font = ImageFont.load_default()
            
            text = "BeyTek"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            
            image_width, image_height = perturbed_image.size
            x = 10
            y = image_height - text_height - 10

            draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)
            
            perturbed_image = Image.alpha_composite(perturbed_image.convert("RGBA"), watermark)
            
            cv2.imwrite(output_filename, cv2.cvtColor(np.array(perturbed_image), cv2.COLOR_RGB2BGR))
            st.image(perturbed_image, caption="Perturbed Image", use_column_width=True)
            st.write(f"Image saved as {output_filename}")
