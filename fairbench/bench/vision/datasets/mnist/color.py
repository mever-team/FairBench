"""
This file contains code originally licensed under the MIT License:

ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python implementation of Biased-MNIST.

Modifications to this code have been made by:
- Emmanouil Krasanakis, © 2024.

Modifications consist of only source code remodularization
and logging behavior without altering base functionality.
"""

# import logging

import numpy as np
import torch
from fairbench.bench.vision.datasets.downloaders import TwoCropTransform
from torch.utils import data
from torchvision import transforms
from fairbench.bench.vision.datasets.mnist.biased import BiasedMNIST


class BiasedMNISTColor(BiasedMNIST):
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
        super(BiasedMNISTColor, self).__init__(
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
        fg_data = torch.zeros_like(data)
        fg_pixel_indices = data != 0
        fg_data[fg_pixel_indices] = 255

        bg_data = torch.zeros_like(data)
        bg_pixel_indices_0 = data == 0
        bg_data[bg_pixel_indices_0] = 1

        bg_data = bg_data * torch.ByteTensor(colour).view(1, 1, 1, 3)

        data = fg_data + bg_data
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
        data = fg_data + bg_data
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
    # logging.info(
    #    f"get_color_mnist - split: {split}, aug: {aug}, given_y: {given_y}, ratio: {ratio}"
    # )
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
        dataset = BiasedMNISTColor(
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
        dataset = BiasedMNISTColor(
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
