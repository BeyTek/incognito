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
    page_title="Icognito BeyTek",
    page_icon="🤐",
)

# Place your PyTorch model loading code here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.mobilenet_v3_small()
model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = True

epsilon = 0.02
image = None
perturbed_image = None

st.title("Adversarial Attack By BeyTek")

# Create a file uploader widget
uploaded_image = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])

# Create a slider widget for attack intensity
epsilon_slider = st.slider("Attack Intensity", min_value=2, max_value=40, value=6)

# Create a button to trigger the attack
if st.button("Attack Image"):
    if uploaded_image is not None:
        # Handle the image upload and attack here
        num_classes = 2
        target = torch.tensor([random.randint(0, num_classes - 1)]).to(device)
        
        # Load the uploaded image and convert it to the appropriate format
        image = Image.open(uploaded_image)
        image = image.convert('RGB')
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(device)

        image_tensor.requires_grad = True
        output = model(image_tensor)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = image_tensor.grad.data
        data_grad = data_grad.detach()
        sign_data_grad = data_grad.sign()
        perturbed_image = image_tensor + epsilon_slider / 100.0 * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = torchvision.transforms.ToPILImage()(perturbed_image.squeeze(0).cpu())

        if perturbed_image:
            perturbed_image_np = np.array(perturbed_image)
            random_uuid = str(uuid.uuid1())[:6]
            output_filename = f"{random_uuid}-Atk.jpg"
            
            # Ajouter un watermark
            watermark = Image.new("RGBA", perturbed_image.size)
            draw = ImageDraw.Draw(watermark)
            font = ImageFont.load_default()  # Utilise la police par défaut de Pillow
            
            text = "BeyTek"
            text_width, text_height = draw.textsize(text, font=font)
            image_width, image_height = perturbed_image.size
            
            x = 10  # Marge de gauche
            y = image_height - text_height - 10  # Marge du bas
            
            # Calculer la boîte englobante du texte
            text_bbox = draw.textbbox((x, y), text, font=font)

            # Placer le texte en bas à gauche
            draw.text((x, y), text, fill=(255, 255, 255, 128), font=font)
            
            perturbed_image = Image.alpha_composite(perturbed_image.convert("RGBA"), watermark)
            
            cv2.imwrite(output_filename, cv2.cvtColor(np.array(perturbed_image), cv2.COLOR_RGB2BGR))
            st.image(perturbed_image, caption="Perturbed Image", use_column_width=True)
            st.write(f"Image saved as {output_filename}")
            
            # Supprimer l'image après 5 secondes
            def delete_image():
                time.sleep(5)  # Attendre 5 secondes
                os.remove(output_filename)  # Supprimer l'image générée
            
            # Créer un bouton de téléchargement avec une action personnalisée
            def on_download():
                with open(output_filename, "rb") as f:
                    data = f.read()
                st.download_button(label="Download Image", data=data, file_name=output_filename, key="download_button")
                thread = threading.Thread(target=delete_image)
                thread.start()
            
            on_download()
        else:
            st.write("Failed to create perturbed image.")
    else:
        st.write("Choose an image first.")

st.markdown("---")
st.title("Credits")

st.write("Made by [BeyTek]")
st.write("[Soutenir](https://ko-fi.com/beytek)")

