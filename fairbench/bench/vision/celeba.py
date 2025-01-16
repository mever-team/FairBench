def celeba(
    classifier="flac",
    data_root=None,
    predict="predict",
    device=None,
    target_attr="blonde",
    batch_size=64,
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
            model_file="celeba_blonde.pth",
            model_url="https://drive.google.com/uc?id=1lIBjjUtZxl3cwa4YuNsrVpp4WqP3jsb5",
            model_class=ResNet18,
        ),
        "mavias": lambda device: get_model(
            device=device,
            model_dir=cache("pretrained/mavias"),
            model_file="celeba_blonde.pth",
            model_url="https://drive.google.com/uc?id=1QBkL8MD9sn8JdkG2ckemUWLWPjR2m-WK",
            model_class=ResNet18,
        ),
    }
    if data_root is None:
        data_root = cache("data/celeba_biased")
    test_loader = get_vision_dataset("celeba")(
        data_root,
        batch_size=batch_size,
        target_attr=target_attr,
        split="valid",
        aug=False,
    )
    return run_dataset(classifiers, test_loader, classifier, predict, device)
