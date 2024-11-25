import os

import torch
from models.resnet import ResNet18
from models.resnet import BAddResNet50
from torchvision.models.resnet import resnet50
import gdown
from torch import nn


def load_celeba_flac_model(device):
    model_dir = "./pretrained/flac"
    model_file = "celeba_blonde.pth"
    model_path = os.path.join(model_dir, model_file)

    # URL to download the model file if it's missing
    model_url = "https://drive.google.com/uc?id=1lIBjjUtZxl3cwa4YuNsrVpp4WqP3jsb5"

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    # Load the model
    model = ResNet18()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["model"]
    )
    model = model.to(device)
    return model


def load_celeba_badd_model(device):
    raise NotImplementedError("The BAdd model for CelebA is not implemented yet.")


def load_celeba_mavias_model(device):
    model_dir = "./pretrained/mavias"
    model_file = "celeba_blonde.pth"
    model_path = os.path.join(model_dir, model_file)

    # URL to download the model file if it's missing
    model_url = "https://drive.google.com/uc?id=1QBkL8MD9sn8JdkG2ckemUWLWPjR2m-WK"

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    # Load the model
    model = ResNet18()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["model"]
    )
    model = model.to(device)
    return model


def load_utkface_flac_model(device):
    model_dir = "./pretrained/flac"
    model_file = "utkface_race.pth"
    model_path = os.path.join(model_dir, model_file)

    # URL to download the model file if it's missing
    model_url = "https://drive.google.com/uc?id=1MToyLcW89IU2G_p7UDYfRZihlllVL4ts"

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    # Load the model
    model = ResNet18()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["model"]
    )
    model = model.to(device)
    return model


def load_utkface_mavias_model(device):
    raise NotImplementedError("The MAVias model for UTKFace is not implemented yet.")


def load_utkface_badd_model(device):
    model_dir = "./pretrained/badd"
    model_file = "utkface_race.pth"
    model_path = os.path.join(model_dir, model_file)

    # URL to download the model file if it's missing
    model_url = "https://drive.google.com/uc?id=1SL_AGaUaxI_NziWFjsRjBY56ROAsa8qN"
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    # Load the model
    model = ResNet18()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["model"]
    )
    model = model.to(device)
    return model


def load_waterbirds_mavias_model(device):
    model_dir = "./pretrained/mavias"
    model_file = "waterbirds.pt"
    model_path = os.path.join(model_dir, model_file)

    # URL to download the model file if it's missing
    model_url = "https://drive.google.com/uc?id=1N5bz67XwkjdC1nliA6onGDt-mSNepNeP"

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    # Load the model
    model = resnet50()
    model.fc = nn.Linear(2048, 2)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["model"]
    )
    model = model.to(device)
    return model


def load_waterbirds_flac_model(device):
    raise NotImplementedError("The FLAC model for Waterbirds is not implemented yet.")


def load_waterbirds_badd_model(device):
    model_dir = "./pretrained/badd"
    model_file = "waterbirds.pth"
    model_path = os.path.join(model_dir, model_file)

    # URL to download the model file if it's missing
    model_url = "https://drive.google.com/uc?id=1jGPmgAVZuwUdsL1CcbsIrdeJqJ9DwXbL"
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    # Load the model
    model = torch.load(model_path, map_location=torch.device(device))
    model = model.to(device)
    return model


def load_imagenet9_mavias_model(device):
    model_dir = "./pretrained/mavias"
    model_file = "imagenet9.pt"
    model_path = os.path.join(model_dir, model_file)

    # URL to download the model file if it's missing
    model_url = "https://drive.google.com/uc?id=1X6rFM5___K3wpQx35e7Q4mZLzpatR9wH"
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Downloading...")
        try:
            # Download the model file from Google Drive
            gdown.download(model_url, model_path, quiet=False)
            print(f"Downloaded model to {model_path}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            raise

    # Load the model
    model = resnet50()
    model.fc = nn.Linear(2048, 9)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device))["model"]
    )
    model = model.to(device)
    return model


def load_imagenet9_flac_model(device):
    raise NotImplementedError("The FLAC model for ImageNet9 is not implemented.")


def load_imagenet9_badd_model(device):
    raise NotImplementedError("The BAdd model for ImageNet9 is not implemented.")
