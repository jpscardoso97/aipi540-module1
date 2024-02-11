# Main script for the project
import streamlit as st
from PIL import Image, ImageDraw
import os
import torch
import cv2
import numpy as np
import supervision as sv
import subprocess
import urllib.request
import time
import matplotlib.pyplot as plt

# Main script for the project
def main():
    st.title('Snake Image Classification')
    st.write('Upload an Image')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Ask user if they would like to presegment for display
        presegmentation = st.radio("Perform Pre-Segmentation using GroundingDINO?", ("Yes", "No"), index = None)
        
        if presegmentation == "Yes":
            from groundingdino.util.inference import load_model, load_image, predict #, annotate, Model

            # Check if CUDA is available
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = "cpu"
            HOME = os.getcwd()
            CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
            WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
            WEIGHTS_PATH = os.path.join(HOME, "GroundingDINO", "weights", WEIGHTS_NAME)

            model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=device)
            GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
            print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))

            GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "GroundingDINO", "weights", WEIGHTS_NAME)
            print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

            
            #grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=device)
            
            TEXT_PROMPT = "snake"
            BOX_TRESHOLD = 0.25
            TEXT_TRESHOLD = 0.25

            # Display bounding box around snake in image
            start_time = time.time()
            _, image_np = load_image(uploaded_file)
            boxes, _, phrases = predict(
                model=model,
                image=image_np,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device = device
            )
            end_time = time.time()
            st.write(f"Segmentation time: {end_time-start_time:.2f} seconds")

            # Draw bounding boxes on the image
            
        elif presegmentation == "No":
            print("no")

# Does not work at the moment
def draw_boxes(image, boxes):
    bounded_image = image.copy()
    draw = ImageDraw.Draw(bounded_image)
    width, height = bounded_image.size

    for box in boxes:
        xmin, ymin, width, height = box
        xmax = xmin + width
        ymax = ymin + height
        x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        
    return bounded_image

def dinoInstaller():
    HOME = os.getcwd()
    # Install GroundingDINO from github if it isn't already present
    if not os.path.isdir(os.path.join(HOME, "GroundingDINO")):
        print("Installing GroundingDINO for Image Segmentation")
        subprocess.run(["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"])
        subprocess.run(["pip", "install", "-q", "-e", "./GroundingDINO"])

    # Create a weights directory if it doesn't exist
    if not os.path.isdir(os.path.join(HOME, "GroundingDINO", "weights")):
        print("GroundingDINO Weights Not Found, Retreiving Weights")
        os.makedirs(os.path.join(HOME, "GroundingDINO", "weights"))
        os.chdir(os.path.join(HOME, "GroundingDINO", "weights"))
        
        # Download the pre-trained model weights using urllib
        urllib.request.urlretrieve("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", "groundingdino_swint_ogc.pth")
    os.chdir(HOME)
    
    print("GroundingDINO Already Installed")

if __name__ == '__main__':
    
    #dinoInstaller()
    main()
