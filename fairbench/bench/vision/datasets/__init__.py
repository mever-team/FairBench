def get_vision_dataset(name):
    from fairbench.bench.vision.datasets.celeba import get_celeba
    from fairbench.bench.vision.datasets.utk_face import get_utk_face
    from fairbench.bench.vision.datasets.waterbirds import get_waterbirds
    import torch

    options = {
        "celeba": get_celeba,
        "utk_face": get_utk_face,
        "waterbirds": get_waterbirds,
    }

    if isinstance(name, torch.nn.Module):
        return name

    if name not in options:
        raise RuntimeError(
            f"The vision dataset {name} does not exist. Please choose one among {list(options.keys())}"
        )

    return options[name]
