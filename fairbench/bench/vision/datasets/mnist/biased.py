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
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fairbench.bench.vision.datasets.downloaders import (
    get_confusion_matrix,
    get_unsup_confusion_matrix,
)
from torchvision.datasets import MNIST


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
                # logging.info(f"load bias feature: {bias_feature_dir}")
                self.bias_features = torch.load(f"{bias_feature_dir}/bias_feats.pt")
                self.marginal = torch.load(f"{bias_feature_dir}/marginal.pt")
            else:
                bias_feature_dir = f"{bias_feature_root}/color_mnist-corrA{data_label_correlation1}-corrB{data_label_correlation2}-seed{seed}"
                # logging.info(f"load bias feature: {bias_feature_dir}")
                self.bias_features = torch.load(f"{bias_feature_dir}/bias_feats.pt")
                self.marginal = torch.load(f"{bias_feature_dir}/marginal.pt")

        save_path = (
            Path(root)
            / "pickles"
            / f"color_mnist-corrA{data_label_correlation1}-corrB{data_label_correlation2}-seed{seed}"
            / split
        )
        if save_path.is_dir():
            # logging.info(f"use existing color_mnist from {save_path}")
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

            # logging.info(f"save color_mnist to {save_path}")
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
