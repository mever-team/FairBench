from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import torch
from datasets.utils import download_imagenet9


def get_imagenet9(root, bench, batch_size=64, workers=4) -> None:
    scale = 256.0 / 224.0
    target_resolution = (224, 224)
    transform_test = transforms.Compose(
        [
            transforms.Resize(
                (
                    int(target_resolution[0] * scale),
                    int(target_resolution[1] * scale),
                )
            ),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if not os.path.isdir(os.path.join(root, "bg_challenge")):
        download_imagenet9(root)

    test_dataset = ImageFolder(
        root=os.path.join(root, "bg_challenge", bench, "val"),
        transform=transform_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return test_loader
