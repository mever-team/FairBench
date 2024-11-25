"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python implementation of Biased-MNIST.
"""
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets.utils import (
    TwoCropTransform,
    get_confusion_matrix,
    get_unsup_confusion_matrix,
)
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import MNIST
import random


class BiasedMNIST(MNIST):
    """A base class for Biased-MNIST.
    We manually select ten colours to synthetic colour bias. (See `COLOUR_MAP` for the colour configuration)
    Usage is exactly same as torchvision MNIST dataset class.

    You have two paramters to control the level of bias.

    Parameters
    ----------
    root : str
        path to MNIST dataset.
    data_label_correlation : float, default=1.0
        Here, each class has the pre-defined colour (bias).
        data_label_correlation, or `rho` controls the level of the dataset bias.

        A sample is coloured with
            - the pre-defined colour with probability `rho`,
            - coloured with one of the other colours with probability `1 - rho`.
              The number of ``other colours'' is controlled by `n_confusing_labels` (default: 9).
        Note that the colour is injected into the background of the image (see `_binary_to_colour`).

        Hence, we have
            - Perfectly biased dataset with rho=1.0
            - Perfectly unbiased with rho=0.1 (1/10) ==> our ``unbiased'' setting in the test time.
        In the paper, we explore the high correlations but with small hints, e.g., rho=0.999.

    n_confusing_labels : int, default=9
        In the real-world cases, biases are not equally distributed, but highly unbalanced.
        We mimic the unbalanced biases by changing the number of confusing colours for each class.
        In the paper, we use n_confusing_labels=9, i.e., during training, the model can observe
        all colours for each class. However, you can make the problem harder by setting smaller n_confusing_labels, e.g., 2.
        We suggest to researchers considering this benchmark for future researches.
    """

    COLOUR_MAP = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [225, 225, 0],
        [225, 0, 225],
        [0, 255, 255],
        [255, 128, 0],
        [255, 0, 128],
        [128, 0, 255],
        [128, 128, 128],
    ]

    COLOUR_MAP2 = [
        [255, 50, 50],
        [50, 255, 50],
        [50, 50, 255],
        [225, 225, 50],
        [225, 50, 225],
        [50, 255, 255],
        [255, 128, 50],
        [255, 50, 128],
        [128, 50, 255],
        [50, 50, 50],
    ]

    def __init__(
        self,
        root,
        bias_feature_root="./biased_feats",
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        data_label_correlation1=1.0,
        data_label_correlation2=1.0,
        n_confusing_labels=9,
        seed=1,
        load_bias_feature=False,
        train_corr=None,
    ):
        assert split in ["train", "valid"]
        train = split in ["train"]
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        self.load_bias_feature = load_bias_feature
        if self.load_bias_feature:
            if train_corr:
                bias_feature_dir = f"{bias_feature_root}/train{train_corr}-corrA{data_label_correlation1}-corrB{data_label_correlation2}-seed{seed}"
                logging.info(f"load bias feature: {bias_feature_dir}")
                self.bias_features = torch.load(f"{bias_feature_dir}/bias_feats.pt")
                self.marginal = torch.load(f"{bias_feature_dir}/marginal.pt")
            else:
                bias_feature_dir = f"{bias_feature_root}/color_mnist-corrA{data_label_correlation1}-corrB{data_label_correlation2}-seed{seed}"
                logging.info(f"load bias feature: {bias_feature_dir}")
                self.bias_features = torch.load(f"{bias_feature_dir}/bias_feats.pt")
                self.marginal = torch.load(f"{bias_feature_dir}/marginal.pt")

        save_path = (
            Path(root)
            / "pickles"
            / f"color_mnist-corrA{data_label_correlation1}-corrB{data_label_correlation2}-seed{seed}"
            / split
        )
        if save_path.is_dir():
            logging.info(f"use existing color_mnist from {save_path}")
            self.data = pickle.load(open(save_path / "data.pkl", "rb"))
            self.targets = pickle.load(open(save_path / "targets.pkl", "rb"))
            self.biased_targets = pickle.load(
                open(save_path / "biased_targets.pkl", "rb")
            )
            self.biased_targets2 = pickle.load(
                open(save_path / "biased_targets2.pkl", "rb")
            )
        else:
            self.random = True

            self.data_label_correlation1 = data_label_correlation1
            self.data_label_correlation2 = data_label_correlation2
            self.n_confusing_labels = n_confusing_labels
            self.biased_targets = torch.zeros_like(self.targets)
            self.biased_targets2 = torch.zeros_like(self.targets)
            self.data = self.data.unsqueeze(-1).expand(-1, -1, -1, 3)

            self.build_biased_mnist("bg")

            self.build_biased_mnist("fg")
            # self.biased_targets2 = self.biased_targets
            indices = np.arange(len(self.data))
            self._shuffle(indices)

            self.data = self.data[indices].numpy()
            self.targets = self.targets[indices]
            self.biased_targets = self.biased_targets[indices]
            self.biased_targets2 = self.biased_targets2[indices]

            logging.info(f"save color_mnist to {save_path}")
            save_path.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.data, open(save_path / "data.pkl", "wb"))
            pickle.dump(self.targets, open(save_path / "targets.pkl", "wb"))
            pickle.dump(
                self.biased_targets, open(save_path / "biased_targets.pkl", "wb")
            )
            pickle.dump(
                self.biased_targets2, open(save_path / "biased_targets2.pkl", "wb")
            )

        if load_bias_feature:
            (
                self.confusion_matrix_org,
                self.confusion_matrix,
            ) = get_unsup_confusion_matrix(
                num_classes=10,
                targets=self.targets,
                biases=self.biased_targets,
                marginals=self.marginal,
            )
        else:
            (
                self.confusion_matrix_org,
                self.confusion_matrix,
                self.confusion_matrix_by,
            ) = get_confusion_matrix(
                num_classes=10, targets=self.targets, biases=self.biased_targets
            )
            (
                self.confusion_matrix_org2,
                self.confusion_matrix2,
                self.confusion_matrix_by2,
            ) = get_confusion_matrix(
                num_classes=10, targets=self.targets, biases=self.biased_targets2
            )

    @property
    def raw_folder(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "processed")

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def _make_biased_mnist(self, indices, label, attribute):
        raise NotImplementedError

    def _update_bias_indices(self, bias_indices, label, data_label_correlation):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * data_label_correlation)
        n_decorrelated_per_class = int(
            np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels))
        )

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(
            indices[n_correlated_samples:], n_decorrelated_per_class
        )

        other_labels = [
            _label % 10
            for _label in range(label + 1, label + 1 + self.n_confusing_labels)
        ]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def build_biased_mnist(self, attribute="fg"):
        """Build biased MNIST."""
        n_labels = self.targets.max().item() + 1

        bias_indices = {label: torch.LongTensor() for label in range(n_labels)}
        if attribute == "fg":
            data_label_correlation = self.data_label_correlation1
        elif attribute == "bg":
            data_label_correlation = self.data_label_correlation2
        for label in range(n_labels):
            self._update_bias_indices(bias_indices, label, data_label_correlation)
        cloned_data = self.data.clone()
        for bias_label, indices in bias_indices.items():
            _data, _ = self._make_biased_mnist(indices, bias_label, attribute)
            cloned_data[indices] = _data
            if attribute == "bg":
                self.biased_targets[indices] = torch.LongTensor(
                    [bias_label] * len(indices)
                )
            else:
                self.biased_targets2[indices] = torch.LongTensor(
                    [bias_label] * len(indices)
                )
        self.data = cloned_data
        return

    def __getitem__(self, index):
        img, target, bias, bias2 = (
            self.data[index],
            int(self.targets[index]),
            int(self.biased_targets[index]),
            int(self.biased_targets2[index]),
        )
        img = Image.fromarray(img.astype(np.uint8), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.load_bias_feature:
            bias_feat = self.bias_features[index]
            return img, target, bias, bias2, index, bias_feat
        else:
            return img, target, bias, bias2, index


class ColourBiasedMNIST(BiasedMNIST):
    def __init__(
        self,
        root,
        bias_feature_root="./biased_feats",
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        data_label_correlation1=1.0,
        data_label_correlation2=1.0,
        n_confusing_labels=9,
        seed=1,
        load_bias_feature=False,
        train_corr=None,
    ):
        super(ColourBiasedMNIST, self).__init__(
            root,
            bias_feature_root=bias_feature_root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
            data_label_correlation1=data_label_correlation1,
            data_label_correlation2=data_label_correlation2,
            n_confusing_labels=n_confusing_labels,
            seed=seed,
            load_bias_feature=load_bias_feature,
            train_corr=train_corr,
        )

    def _binary_to_colour(self, data, colour):
        # fg_data = torch.zeros_like(data)
        # fg_data[data != 0] = 255
        # fg_data[data == 0] = 0
        # fg_data = torch.stack([fg_data, fg_data, fg_data], dim=1)
        fg_data = torch.zeros_like(data)
        fg_pixel_indices = data != 0
        # fg_pixel_indices = (
        #     fg_pixel_indices[:, :, :, 0]
        #     & fg_pixel_indices[:, :, :, 1]
        #     & fg_pixel_indices[:, :, :, 2]
        # )
        fg_data[fg_pixel_indices] = 255

        # bg_data = torch.zeros_like(data)
        # bg_data[data == 0] = 1
        # bg_data[data != 0] = 0
        # bg_data = torch.stack([bg_data, bg_data, bg_data], dim=3)

        bg_data = torch.zeros_like(data)
        bg_pixel_indices_0 = data == 0
        # bg_pixel_indices_0 = (
        #     bg_pixel_indices_0[:, :, :, 0]
        #     & bg_pixel_indices_0[:, :, :, 1]
        #     & bg_pixel_indices_0[:, :, :, 2]
        # )
        bg_data[bg_pixel_indices_0] = 1

        # bg_pixel_indices_1 = data != 0
        # # bg_pixel_indices_1 = (
        # #     bg_pixel_indices_1[:, :, :, 0]
        # #     & bg_pixel_indices_1[:, :, :, 1]
        # #     & bg_pixel_indices_1[:, :, :, 2]
        # # )
        # bg_data[bg_pixel_indices_1] = 0

        bg_data = bg_data * torch.ByteTensor(colour).view(1, 1, 1, 3)

        data = fg_data + bg_data
        # first_image = data[0].numpy()  # .transpose(1, 2, 0)
        # import matplotlib.pyplot as plt

        # # Display the image
        # plt.imshow(first_image)
        # plt.savefig("temp.png")
        # plt.show()
        return data

    def _binary_to_colour2(self, data, colour):
        fg_data = torch.zeros_like(data)
        white_pixel_indices = data == 255

        white_pixel_indices = (
            white_pixel_indices[:, :, :, 0]
            & white_pixel_indices[:, :, :, 1]
            & white_pixel_indices[:, :, :, 2]
        )

        fg_data[white_pixel_indices] = 1

        black_pixel_indices = fg_data == 0
        black_pixel_indices = (
            black_pixel_indices[:, :, :, 0]
            & black_pixel_indices[:, :, :, 1]
            & black_pixel_indices[:, :, :, 2]
        )

        bg_data = torch.zeros_like(data)
        bg_data[black_pixel_indices] = data[black_pixel_indices]
        fg_data = fg_data * torch.ByteTensor(colour)
        # print(torch.sum(white_pixel_indices), torch.sum(black_pixel_indices))
        data = fg_data + bg_data
        # import matplotlib.pyplot as plt

        # first_image = data[0].numpy()  # .transpose(1, 2, 0)

        # # Display the image
        # plt.imshow(first_image)
        # plt.savefig("temp.png")
        # plt.show()
        # data[-11111111111111111111]
        return data

    def _make_biased_mnist(self, indices, label, attribute):
        if attribute == "bg":
            return (
                self._binary_to_colour(self.data[indices], self.COLOUR_MAP[label]),
                self.targets[indices],
            )
        elif attribute == "fg":
            return (
                self._binary_to_colour2(
                    self.data[indices],
                    self.COLOUR_MAP2[(label + 1) % 10],
                ),
                self.targets[indices],
            )
        else:
            raise ValueError(attribute)


def get_color_mnist(
    root,
    batch_size,
    data_label_correlation1,
    data_label_correlation2,
    n_confusing_labels=9,
    split="train",
    num_workers=4,
    seed=1,
    aug=True,
    two_crop=False,
    ratio=0,
    bias_feature_root="./biased_feats",
    load_bias_feature=False,
    given_y=True,
    train_corr=None,
):
    logging.info(
        f"get_color_mnist - split: {split}, aug: {aug}, given_y: {given_y}, ratio: {ratio}"
    )
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if aug:
        prob = 0.5
        train_transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.RandomResizedCrop(28, scale=(0.75, 1)),
                    ],
                    p=prob,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(20),
                    ],
                    p=prob,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(20),
                    ],
                    p=prob,
                ),
                # transforms.GaussianBlur(3),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if two_crop:
        train_transform = TwoCropTransform(train_transform)

    if split == "train_val":
        dataset = ColourBiasedMNIST(
            root,
            split="train",
            transform=train_transform,
            download=True,
            data_label_correlation1=data_label_correlation1,
            data_label_correlation2=data_label_correlation2,
            n_confusing_labels=n_confusing_labels,
            seed=seed,
            load_bias_feature=load_bias_feature,
            train_corr=train_corr,
        )

        indices = list(range(len(dataset)))
        split = int(np.floor(0.1 * len(dataset)))
        np.random.shuffle(indices)
        valid_idx = indices[:split]
        valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)

        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return dataloader

    else:
        dataset = ColourBiasedMNIST(
            root,
            bias_feature_root=bias_feature_root,
            split=split,
            transform=train_transform,
            download=True,
            data_label_correlation1=data_label_correlation1,
            data_label_correlation2=data_label_correlation2,
            n_confusing_labels=n_confusing_labels,
            seed=seed,
            load_bias_feature=load_bias_feature,
            train_corr=train_corr,
        )

        def clip_max_ratio(score):
            upper_bd = score.min() * ratio
            # print(upper_bd)
            return np.clip(score, None, upper_bd)

        if ratio != 0:
            if load_bias_feature:
                weights = dataset.marginal
            else:
                if given_y:
                    weights = [
                        1 / dataset.confusion_matrix_by[c, b]
                        for c, b in zip(dataset.targets, dataset.biased_targets)
                    ]
                    weights2 = [
                        1 / dataset.confusion_matrix_by2[c, b]
                        for c, b in zip(dataset.targets, dataset.biased_targets2)
                    ]
                    weights = weights + weights2
                else:
                    weights = [
                        1 / dataset.confusion_matrix[b, c]
                        for c, b in zip(dataset.targets, dataset.biased_targets)
                    ]
                    weights2 = [
                        1 / dataset.confusion_matrix2[b, c]
                        for c, b in zip(dataset.targets, dataset.biased_targets2)
                    ]
                weights = [max(w1, w2) for w1, w2 in zip(weights, weights2)]

            if ratio > 0:
                weights = clip_max_ratio(np.array(weights))
            sampler = data.WeightedRandomSampler(
                weights, len(weights), replacement=True
            )
        else:
            sampler = None

        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True if sampler is None and split == "train" else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=split == "train",
        )

        return dataloader
