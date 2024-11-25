def get_vision_dataset(name):
    from fairbench.bench.vision.datasets.mnist import BiasedMNIST, BiasedMNISTColor, BiasedMNISTSingle
    from fairbench.bench.vision.datasets.celeba import get_celeba
    from fairbench.bench.vision.datasets.utk_face import UTKFace, BiasedUTKFace
    from fairbench.bench.vision.datasets.waterbirds import WaterbirdsDataset
    from fairbench.bench.vision.datasets.imagenet9 import get_imagenet9

    options = {
        "biased_mnist": BiasedMNIST,
        "biased_mnist_color": BiasedMNISTColor,
        "biased_mnist_single": BiasedMNISTSingle,
        "biased_celeba": get_celeba,
        "utk_face": UTKFace,
        "imagenet9": get_imagenet9
    }

    if name not in options:
        raise RuntimeError(f"The vision dataset {name} does not exist. Please choose one among {list(options.keys())}")

    return options[name]