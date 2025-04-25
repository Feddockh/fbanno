import os
import subprocess
import sys
import zipfile
import gdown
import shutil
import re


# ----- Configuration Variables -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DINOv2 repository configuration
# DINOV2_REPO_URL = "https://github.com/facebookresearch/dinov2.git"
# DINOV2_DIR = "dinov2"

# Rivendale dataset configuration
RIVENDALE_ZIP_URL = "https://drive.google.com/file/d/14hANjlKxG5DKPuh1oDa4T8PoYJ1J6Uwr/view?usp=drive_link"
RIVENDALE_DIR = "rivendale_dataset"

# Erwiand dataset configuration
ERWIAM_ZIP_URL = "https://drive.google.com/file/d/19TJcN04cpvpbew-htK-Z0UFLf942bU2f/view?usp=drive_link"
ERWIAM_DIR = "erwiam_dataset"

# Foundation Stereo checkpoint configuration
CHECKPOINT_ZIP_URL = "https://drive.google.com/file/d/1zrrvLcpJEYHiQYJXTDabhXz4YWI6zmD8/view?usp=drive_link"
CHECKPOINT_DIR = "FoundationStereo/pretrained_models/23-51-11"


def clone_repo(repo_url: str, target_dir: str) -> None:
    """
    Clone a git repository from a given URL into the specified target directory.
    Skips cloning if the directory already exists.
    """
    if os.path.exists(os.path.join(BASE_DIR, target_dir)):
        print(f"Directory '{target_dir}' already exists. Skipping clone.")
        return

    print(f"Cloning repository from {repo_url} into '{target_dir}'")
    try:
        subprocess.check_call(["git", "clone", repo_url, os.path.join(BASE_DIR, target_dir)])
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while cloning the repository: {e}")
        sys.exit(1)

def download_checkpoint(cfg_url: str, pth_url: str, target_dir: str) -> None:
    """
    Download a cfg.yaml file and a .pth file from Google Drive and place them inside the target directory.
    
    If the target directory already exists, the download is skipped.
    
    Args:
        cfg_url (str): The Google Drive URL of the cfg.yaml file.
        pth_url (str): The Google Drive URL of the .pth file.
        target_dir (str): The directory where the files will be downloaded.
    """
    download_dir = os.path.join(BASE_DIR, target_dir)
    if os.path.exists(download_dir):
        print(f"Checkpoint directory '{target_dir}' already exists. Skipping download.")
        return

    os.makedirs(download_dir, exist_ok=True)

    # Download the cfg.yaml file.
    cfg_output = os.path.join(download_dir, "cfg.yaml")
    print(f"Downloading cfg.yaml from Google Drive: {cfg_url}")
    try:
        gdown.download(cfg_url, cfg_output, quiet=False, fuzzy=True)
        print("cfg.yaml downloaded successfully.")
    except Exception as e:
        print(f"Error occurred during cfg.yaml download: {e}")
        sys.exit(1)

    # Download the .pth file.
    pth_output = os.path.join(download_dir, "model.pth")
    print(f"Downloading model.pth from Google Drive: {pth_url}")
    try:
        gdown.download(pth_url, pth_output, quiet=False, fuzzy=True)
        print("model.pth downloaded successfully.")
    except Exception as e:
        print(f"Error occurred during model.pth download: {e}")
        sys.exit(1)

def download_and_extract_google_drive_zip(zip_url: str, target_dir: str) -> None:
    """
    Download a zip file from a Google Drive URL and extract its contents.
    
    If the extraction directory already exists, the download and extraction are skipped.
    After extraction, the zip file is removed.
    
    Args:
        zip_url (str): The Google Drive URL of the zip file.
        target_dir (str): The directory where the zip file will be extracted.
    """
    zip_filename = target_dir + ".zip"
    zip_path = os.path.join(BASE_DIR, zip_filename)
    extract_dir = os.path.join(BASE_DIR, target_dir)

    if os.path.exists(extract_dir):
        print(f"Data directory '{target_dir}' already exists. Skipping download and extraction.")
        return

    # Download the zip file if it doesn't exist.
    if not os.path.exists(zip_path):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        print(f"Downloading zip file from Google Drive: {zip_url}")
        gdown.download(zip_url, zip_path, quiet=False, fuzzy=True)
    else:
        print(f"Zip file '{zip_filename}' already exists. Skipping download.")

    # Extract the zip file.
    print(f"Extracting '{zip_filename}' into '{target_dir}'")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            top_level_ref = zip_ref.namelist()[0]
            if top_level_ref != os.path.basename(target_dir) + '/':
                print(f"Error: The top level ref '{top_level_ref}' does not match the target dir.")
            else:
                zip_ref.extractall(os.path.dirname(extract_dir))
                print(f"Extracted contents to '{BASE_DIR}'")
    except zipfile.BadZipFile as e:
        print(f"Error extracting {zip_filename}: {e}")
        sys.exit(1)

    # Delete the zip file after extraction.
    os.remove(zip_path)
    print(f"Deleted zip file '{zip_filename}' after extraction.")

def main():
    """
    Main function to set up the environment by cloning repositories and downloading datasets.
    """
    # Clone the DINOv2 repository
    # clone_repo(DINOV2_REPO_URL, DINOV2_DIR)

    # Download the Foundation Stereo checkpoint
    download_and_extract_google_drive_zip(CHECKPOINT_ZIP_URL, CHECKPOINT_DIR)

    # Download and extract the Rivendale dataset
    download_and_extract_google_drive_zip(RIVENDALE_ZIP_URL, RIVENDALE_DIR)

    # # Download and extract the Erwiand dataset
    # download_and_extract_google_drive_zip(ERWIAM_ZIP_URL, ERWIAM_DIR)

if __name__ == "__main__":
    main()