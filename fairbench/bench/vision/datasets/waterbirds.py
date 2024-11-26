import csv
import os
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from fairbench.bench.vision.datasets.downloaders import download_waterbirds

data_split = {0: "train", 1: "val", 2: "test"}


class WaterbirdsDataset(Dataset):
    def __init__(
        self,
        raw_data_path,
        root,
        split="train",
        transform=None,
        target_transform=None,
        return_places=False,
    ) -> None:
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_places = return_places
        self.return_masked = False
        img_data_dir = os.path.join(root, "images", split)

        if not os.path.isdir(os.path.join(root, "images", "test")):
            download_waterbirds(root)
        self.places = {}
        with open(os.path.join(root, "metadata.csv")) as meta_file:
            csv_reader = csv.reader(meta_file)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                img_id, img_filename, y, split_index, place, place_filename = row
                if data_split[int(split_index)] == split:
                    self.places[img_filename.split("/")[-1]] = int(place)
        self.update_data(img_data_dir)

    def update_data(self, data_file_directory, masked_data_file_path=None):
        self.data_path = []
        self.masked_data_path = []
        self.targets = []
        data_classes = sorted(os.listdir(data_file_directory))
        print("-" * 10, f"indexing {self.split} data", "-" * 10)
        for data_class in tqdm(data_classes):
            target = int(data_class)
            class_image_file_paths = glob(
                os.path.join(data_file_directory, data_class, "*")
            )
            self.data_path += class_image_file_paths
            if masked_data_file_path is not None:
                self.return_masked = True
                masked_class_image_file_paths = sorted(
                    glob(os.path.join(masked_data_file_path, data_class, "*"))
                )
                self.masked_data_path += masked_class_image_file_paths
            self.targets += [target] * len(class_image_file_paths)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_file_path, target)
        """
        img_file_path, target = self.data_path[index], self.targets[index]
        img = Image.open(img_file_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, img_file_path, target, self.places[img_file_path.split("/")[-1]]


def get_waterbirds(root_dir, batch_size=64, n_workers=4) -> None:
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

    test_dataset = WaterbirdsDataset(
        raw_data_path=root_dir,
        root=root_dir,
        split="test",
        transform=transform_test,
        return_places=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )
    return test_loader
