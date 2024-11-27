import os
import torch
import gdown


def _download_model(model_url, model_path):
    print(f"Model file not found at {model_path}. Downloading...")
    try:
        gdown.download(model_url, model_path, quiet=False)
        print(f"Downloaded model to {model_path}.")
    except Exception as e:
        print(f"Error downloading the model: {e}")
        raise


def get_model(
    device, model_dir, model_file, model_url, model_class, model_in_state_dict="model"
):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_file)

    if not os.path.exists(model_path):
        _download_model(model_url, model_path)

    state_dict = torch.load(
        model_path, map_location=torch.device(device), weights_only=False
    )
    if model_in_state_dict is not None:
        state_dict = state_dict[model_in_state_dict]
    model = model_class()
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model
