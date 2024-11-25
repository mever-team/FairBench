# import gdown
import tarfile
import os

from glob import glob
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch
from torchvision import transforms
from datasets.utils import (
    TwoCropTransform,
    get_confusion_matrix,
    get_unsup_confusion_matrix,
)
import numpy as np
from torch.utils import data


class CorruptedCIFAR10(Dataset):
    def __init__(self, root, split, percent, transform=None, image_path_list=None):
        super().__init__()

        if not os.path.isdir(os.path.join(root, "cifar10c")):
            self.download_dataset(root)
        root = os.path.join(root, "cifar10c", percent)

        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split == "train":
            self.align = glob(os.path.join(root, "align", "*", "*"))
            self.conflict = glob(os.path.join(root, "conflict", "*", "*"))
            self.data = self.align + self.conflict

        elif split == "valid":
            self.data = glob(os.path.join(root, split, "*", "*"))

        elif split == "test":
            self.data = glob(os.path.join(root, "../test", "*", "*"))

        self.set_targets_biases()
        (
            self.confusion_matrix_org,
            self.confusion_matrix,
            self.confusion_matrix_by,
        ) = get_confusion_matrix(
            num_classes=10, targets=self.targets, biases=self.biased_targets
        )

    def download_dataset(self, path):
        url = "https://drive.google.com/file/d/1_eSQ33m2-okaMWfubO7b8hhvLMlYNJP-/view?usp=sharing"
        output = os.path.join(path, "cifar10c.tar.gz")
        print(f"=> Downloading corrupted CIFAR10 dataset from {url}")
        # gdown.download(url, output, quiet=False, fuzzy=True)

        print("=> Extracting dataset..")
        tar = tarfile.open(os.path.join(path, "cifar10c.tar.gz"), "r:gz")
        tar.extractall(path=path)
        tar.close()
        os.remove(output)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, bias = int(self.data[index].split("_")[-2]), int(
            self.data[index].split("_")[-1].split(".")[0]
        )
        image = Image.open(self.data[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, bias

    def set_targets_biases(self):
        self.targets = []
        self.biased_targets = []
        for sample in self.data:
            self.targets.append(torch.tensor(int(sample.split("_")[-2])))
            self.biased_targets.append(
                torch.tensor(int(sample.split("_")[-1].split(".")[0]))
            )


def get_cifar(opt, twocrop, aug):
    ratio = 0 #100
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    T_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    if aug:
        T_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    T_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    if twocrop:
        T_train = TwoCropTransform(T_train)

    corruption = opt.dataset.replace("corrupted-cifar10_", "")
    opt.corruption = corruption

    train_dataset = CorruptedCIFAR10(
        root=opt.data_dir, split="train", percent=corruption, transform=T_train
    )
    test_dataset = CorruptedCIFAR10(
        root=opt.data_dir, split="test", percent=corruption, transform=T_test
    )

    def clip_max_ratio(score):
        upper_bd = score.min() * ratio
        return np.clip(score, None, upper_bd)

    if ratio != 0:
        weights = [
            1 / train_dataset.confusion_matrix_by[c, b]
            for c, b in zip(train_dataset.targets, train_dataset.biased_targets)
        ]
        if ratio > 0:
            weights = clip_max_ratio(np.array(weights))
        print(max(weights))
        sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        sampler = None

    opt.n_classes = 10
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bs,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    dataset = CorruptedCIFAR10(
        f'/home/{os.environ.get("USER")}/temp', "train", "0.5pct"
    )
