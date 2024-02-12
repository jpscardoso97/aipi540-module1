# Main script for the project
import streamlit as st
from PIL import Image
import os
import torch
import numpy as np
import subprocess
import urllib.request
import time
from torchvision.ops import box_convert
from scripts.inference.predict import Predictor
import sys
current_dir = os.getcwd()
utils_dir = os.path.join(current_dir, 'utils')
sys.path.append(utils_dir)
import class_mapping

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
            from groundingdino.util.inference import load_model, load_image, predict, annotate

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
      
            TEXT_PROMPT = "snake"
            BOX_TRESHOLD = 0.25
            TEXT_TRESHOLD = 0.25

            # Display bounding box around snake in image
            start_time = time.time()
            image_source, image_np_tensor = load_image(uploaded_file)
            
            boxes, logits, _ = predict(
                model=model,
                image=image_np_tensor,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device = device
            )
            end_time = time.time()
            st.write(f"Segmentation time: {end_time-start_time:.2f} seconds")

            cropped_image, logit_confidence = crop_image(image_source, boxes, logits)
            # Display the cropped image
            st.image(cropped_image, caption='Cropped Image', use_column_width=True)
            st.write(f"Snake Segmentation Confidence: {logit_confidence}")

            predict_image_and_display(cropped_image)
            
        elif presegmentation == "No":
            predict_image_and_display(image)

def predict_image_and_display(image):
    predictor = Predictor()
    preds = predictor.predict(image)

    for i, c in enumerate(class_mapping.map_classes()):
        if preds[i] == 1:
            st.markdown("## Class Predictions:")
            st.markdown(f"### {c}")
            return c
    return None

def crop_image(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
    max_index = torch.argmax(logits)

    h, w, _ = image_source.shape
    box = boxes[max_index] * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=box.unsqueeze(0), in_fmt="cxcywh", out_fmt="xyxy").numpy()

    xmin, ymin, xmax, ymax = xyxy.squeeze().astype(int)
    cropped_image = image_source[ymin:ymax, xmin:xmax]

    return cropped_image, logits[max_index]

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
