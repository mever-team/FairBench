from fairbench.bench.vision.datasets import get_vision_dataset
from fairbench.bench.vision.architectures.runner import run_dataset
from fairbench.bench.vision.architectures.loader import get_model


def waterbirds(
    classifier="flac",
    data_root="./data/waterbirds",
    predict="predict",
    device=None,
):
    from torchvision.models.resnet import resnet50
    from fairbench.bench.vision.architectures.resnet import BAddResNet50

    classifiers = {
        "badd": lambda device: get_model(
            device=device,
            model_dir="./pretrained/badd",
            model_file="waterbirds.pth",
            model_url="https://drive.google.com/uc?id=1jGPmgAVZuwUdsL1CcbsIrdeJqJ9DwXbL",
            model_class=BAddResNet50,
        ),
        "mavias": lambda device: get_model(
            device=device,
            model_dir="./pretrained/mavias",
            model_file="waterbirds.pt",
            model_url="https://drive.google.com/uc?id=1N5bz67XwkjdC1nliA6onGDt-mSNepNeP",
            model_class=resnet50,
        ),
    }
    test_loader = get_vision_dataset("waterbirds")(data_root, batch_size=64)
    return run_dataset(classifiers, test_loader, classifier, predict, device)
