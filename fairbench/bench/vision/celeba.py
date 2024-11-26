from fairbench.bench.vision.datasets import get_vision_dataset
from fairbench.bench.vision.architectures.runner import run_dataset
from fairbench.bench.vision.architectures.loader import get_model


def celeba(
    classifier="flac",
    data_root="./data/celeba",
    predict="predict",
    device=None,
):
    from fairbench.bench.vision.architectures.resnet import ResNet18

    classifiers = {
        "flac": lambda device: get_model(
            device=device,
            model_dir="./pretrained/flac",
            model_file="celeba_blonde.pth",
            model_url="https://drive.google.com/uc?id=1lIBjjUtZxl3cwa4YuNsrVpp4WqP3jsb5",
            model_class=ResNet18,
        ),
        "mavias": lambda device: get_model(
            device=device,
            model_dir="./pretrained/mavias",
            model_file="celeba_blonde.pth",
            model_url="https://drive.google.com/uc?id=1QBkL8MD9sn8JdkG2ckemUWLWPjR2m-WK",
            model_class=ResNet18,
        ),
    }
    test_loader = get_vision_dataset("celeba")(
        data_root, batch_size=64, target_attr="blonde", split="valid", aug=False
    )
    return run_dataset(classifiers, test_loader, classifier, predict, device)