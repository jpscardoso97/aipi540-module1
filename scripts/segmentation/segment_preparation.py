import os
import subprocess
import requests
import sys

HOME = os.getcwd()

### Clone the repository and install the package

# Define the repository URL and the directory name for cloning
repo_url = "https://github.com/IDEA-Research/GroundingDINO.git"
repo_dir = "GroundingDINO"
home_dir = HOME

os.chdir(home_dir)

subprocess.run(["git", "clone", repo_url])
os.chdir(os.path.join(home_dir, repo_dir))

# Install the package using subprocess to call the pip command
subprocess.run(["pip", "install", "-q", "-e", "."])

CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))


### Download the model weights

file_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

weights_dir = os.path.join(home_dir, "weights")

# Create the weights directory if it does not exist
os.makedirs(weights_dir, exist_ok=True)

file_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")

# Download the file with a stream to avoid loading it all into memory at once
response = requests.get(file_url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024): 
            # Filter out keep-alive new chunks
            if chunk:
                f.write(chunk)
    print("Download completed successfully!")
else:
    print(f"Download failed with status code: {response.status_code}")
