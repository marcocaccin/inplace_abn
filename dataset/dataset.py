from itertools import chain

import glob
import torch
import s3fs
from PIL import Image
from os import path
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, transform):
        super(SegmentationDataset, self).__init__()

        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self.images = []
        for img_path in chain(*(glob.iglob(path.join(self.in_dir, ext)) for ext in SegmentationDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            self.images.append({
                "idx": idx,
                "path": img_path
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))

        return {"img": img, "meta": {"idx": self.images[item]["idx"], "size": size}}


class S3Dataset(Dataset):
    def __init__(self, bucket_path, transform):
        super(S3Dataset, self).__init__()

        self.path = bucket_path
        self.transform = transform
        self.s3 = s3fs.S3FileSystem(
            s3_additional_kwargs={'ServerSideEncryption': 'AES256'}
        )

        # Find all images
        self.images = []
        for img_path in self.s3.ls(path):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)
            self.images.append({
                "idx": idx,
                "path": img_path
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self.images[item]["path"]) as img_raw:
            size = img_raw.size
            img = self.transform(img_raw.convert(mode="RGB"))

        return {"img": img, "meta": {"idx": self.images[item]["idx"], "size": size}}


def segmentation_collate(items):
    imgs = torch.stack([item["img"] for item in items])
    metas = [item["meta"] for item in items]

    return {"img": imgs, "meta": metas}
