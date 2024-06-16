import os
import cv2
import numpy as np
import uuid
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = torchvision.models.mobilenet_v3_small(pretrained=True).to(DEVICE).eval()
ALPHA_DEFAULT = 0.01
NUM_ITER_DEFAULT = 40

# Streamlit configuration
st.set_page_config(page_title="Incognito BeyTek", page_icon="🤐")
st.title("Adversarial Attack By BeyTek")

# UI Elements
uploaded_image = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])
epsilon_slider = st.slider("Attack Intensity", min_value=2, max_value=40, value=4)

def pgd_attack(image_tensor, epsilon, alpha=ALPHA_DEFAULT, num_iter=NUM_ITER_DEFAULT):
    perturbed_image = image_tensor.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        output = MODEL(perturbed_image)
        criterion = torch.nn.CrossEntropyLoss()
        target = torch.tensor([torch.argmax(output)]).to(DEVICE)
        loss = criterion(output, target)
        MODEL.zero_grad()
        loss.backward()
        perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
        eta = torch.clamp(perturbed_image - image_tensor, -epsilon, epsilon)
        perturbed_image = torch.clamp(image_tensor + eta, 0, 1).detach_()
    return perturbed_image

def save_and_display_image(perturbed_image):
    perturbed_image = torchvision.transforms.ToPILImage()(perturbed_image.squeeze(0).cpu())
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

if st.button("Attack Image"):
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
        image_tensor.requires_grad = True
        
        epsilon = epsilon_slider / 100.0
        perturbed_image = pgd_attack(image_tensor, epsilon)
        
        if perturbed_image is not None:
            save_and_display_image(perturbed_image)
    else:
        st.write("Please upload an image to attack.")

