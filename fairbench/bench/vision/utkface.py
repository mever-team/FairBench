def utkface(
    classifier="flac",
    data_root=None,
    predict="predict",
    device=None,
):
    from fairbench.bench.vision.datasets import get_vision_dataset
    from fairbench.bench.vision.architectures.runner import run_dataset
    from fairbench.bench.vision.architectures.loader import get_model
    from fairbench.bench.loader import cache
    from fairbench.bench.vision.architectures.resnet import ResNet18

    classifiers = {
        "flac": lambda device: get_model(
            device=device,
            model_dir=cache("pretrained/flac"),
            model_file="utkface_race.pth",
            model_url="https://drive.google.com/uc?id=1MToyLcW89IU2G_p7UDYfRZihlllVL4ts",
            model_class=ResNet18,
        ),
        "badd": lambda device: get_model(
            device=device,
            model_dir=cache("pretrained/badd"),
            model_file="utkface_race.pth",
            model_url="https://drive.google.com/uc?id=1SL_AGaUaxI_NziWFjsRjBY56ROAsa8qN",
            model_class=ResNet18,
        ),
    }
    if data_root is None:
        data_root = cache("data/utk_face")
    test_loader = get_vision_dataset("utk_face")(
        data_root, batch_size=64, bias_attr="race", split="test", aug=False
    )
    return run_dataset(classifiers, test_loader, classifier, predict, device)
