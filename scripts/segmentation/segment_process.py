import os
import sys

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import Model


import torch
from torchvision.ops import box_convert
import numpy as np
from PIL import Image
import supervision as sv

import cv2

HOME = os.getcwd()

def load_DINO_model():
  '''
  With the GoundingDINO repository cloned and the model weights downloaded in the same directory as the script,
  this function loads the model and returns it.
  '''

  # Define the path to the repository and change the working directory
  home_dir = HOME
  repo_dir = "GroundingDINO"
  os.chdir(os.path.join(home_dir, repo_dir))

  sys.path.append(os.path.join(home_dir, repo_dir))

  # Load the model
  CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
  print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
  WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
  WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
  print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

  model = load_model(CONFIG_PATH, WEIGHTS_PATH)

  return model

# load the model
model = load_DINO_model()

TEXT_PROMPT = "snake"
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25

def get_bounding_box(path):
  image_source, image = load_image(path)

  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=TEXT_PROMPT,
      box_threshold=BOX_TRESHOLD,
      text_threshold=TEXT_TRESHOLD
  )

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

  return boxes[0].numpy()

def get_cropped_image(img_path, box):

  center_x_pct = box[0]
  center_y_pct = box[1]
  width_pct = box[2]
  height_pct = box[3]

  img = cv2.imread(img_path)

  img_height, img_width, _ = img.shape

  # Convert normalized coordinates to pixel values
  center_x = int(center_x_pct * img_width)
  center_y = int(center_y_pct * img_height)
  width = int(width_pct * img_width)
  height = int(height_pct * img_height)

  print("Center:", (center_x, center_y))
  print("Width:", width)
  print("Height:", height)

  x1 = center_x - width // 2
  x2 = center_x + width // 2
  y1 = center_y - height // 2
  y2 = center_y + height // 2

  cropped_img = img[y1:y2, x1:x2]

  return cropped_img

# iterate over all images in the data/raw folder and subfolders and crop the snake
RAW_DATA_PATH = os.path.join(HOME, "../../data/raw")

flag = 0
for root, dirs, files in os.walk(RAW_DATA_PATH):
    for file in files:
        img_path = os.path.join(root, file)
        print(img_path)
        try:
          box = get_bounding_box(img_path)
          cropped_img = get_cropped_image(img_path, box)

          # Create the processed folder to store the cropped images
          if not os.path.exists(img_path.replace("raw", "processed").rsplit('/', 1)[0]):
            os.makedirs(img_path.replace("raw", "processed").rsplit('/', 1)[0])
          
          cropped_img_path = img_path.replace("raw", "processed")
          cv2.imwrite(cropped_img_path, cropped_img)
          flag += 1
          print('Number of images processed:', flag)
        except:
          print("Error in writing file:", img_path)
          flag += 1
          # save the error in a log file
          with open("error_log.txt", "a") as file:
            file.write("Error in writing file: " + img_path + "\n")
          continue