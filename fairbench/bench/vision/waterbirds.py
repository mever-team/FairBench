def waterbirds(
    classifier="badd",
    data_root=None,
    predict="predict",
    device=None,
):
    from fairbench.bench.vision.datasets import get_vision_dataset
    from fairbench.bench.vision.architectures.runner import run_dataset
    from fairbench.bench.vision.architectures.loader import get_model
    from fairbench.bench.loader import cache
    from torchvision.models.resnet import resnet50
    from fairbench.bench.vision.architectures.resnet import BAddResNet50
    from torch import nn

    def baddresnet50():
        model = BAddResNet50()
        model.fc = nn.Linear(2048, 2)
        return model

    def rn50():
        model = resnet50()
        model.fc = nn.Linear(2048, 2)
        return model

    classifiers = {
        "badd": lambda device: get_model(
            device=device,
            model_dir=cache("pretrained/badd"),
            model_file="waterbirds.pth",
            model_url="https://drive.google.com/uc?id=1BMAis2LSuiQQK7OUn0T4lqKAMuHkfWm1",
            model_class=baddresnet50,
            model_in_state_dict=None,
        ),
        "mavias": lambda device: get_model(
            device=device,
            model_dir=cache("pretrained/mavias"),
            model_file="waterbirds.pt",
            model_url="https://drive.google.com/uc?id=1N5bz67XwkjdC1nliA6onGDt-mSNepNeP",
            model_class=rn50,
        ),
    }
    if data_root is None:
        data_root = cache("data/waterbirds")
    test_loader = get_vision_dataset("waterbirds")(data_root, batch_size=64)
    return run_dataset(
        classifiers, test_loader, classifier, predict, device, unpacking=[0, 2, 3]
    )
