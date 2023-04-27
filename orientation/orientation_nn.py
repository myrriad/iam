import glob
import math

import PIL
import numpy as np
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import os
from typing import Callable, Optional, Tuple, Any, List, Dict, cast

import torchvision.transforms.functional
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension, default_loader

from pathlib import Path

# transform180 =
class OrientationImageFolder(ImageFolder):
    """
    Modified ImageFolder which
    a) uses only one class, so no sub-folders are needed
    b) also applies a transform to get orientation.
    https://github.com/pytorch/vision/blob/b403bfc771e0caf31efd06d43860b09004f4ac61/torchvision/datasets/folder.py#L48
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader, # PIL loader
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    #@override
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        # customize to only use one class
        classes = ['normal', 'flipped']
        class_to_idx = {'normal': 0, 'flipped': 1}
        return classes, class_to_idx

    # TODO: This can be overridden (make_dataset) to e.g. read files from a compressed zip file instead of from the disk.
    # consider: https://github.com/ain-soph/trojanzoo/issues/42
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        # customize to only use one

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]
        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        # for root, _, fnames in sorted(os.walk(directory)):
        if directory.endswith('/') or directory.endswith('\\'):
            directory = directory[:-1]

        for path in glob.iglob(directory + '/**/*.*', recursive=True):
            # print(path)
            # root = directory # what's the difference?

            # for fname in sorted(fnames):
            # path = os.path.join(root, fname)
            if is_valid_file(path):
                item = (path, 0) # we can ignore the class index, since we only have one class
                # class index is always 0
                instances.append(item)
                # instances.append((path, 1)) # insert upside down version as the same.
                # we apply a transform in __getitem__ to flip the image
        return instances

    def __len__(self) -> int:
        return 2 * len(self.samples)

    def __getitem__(self, index: int) -> Any:
        path, _ = self.samples[index // 2]
        sample = self.loader(path)

        target = index % 2
        if target == 1:
            # torchvision.transforms.RandomRotation((180, 180), expand=False)
            # sample = transform180(sample)

            sample = torchvision.transforms.functional.rotate(sample, 180, expand=False)

            # i'm desperate. just invert the whole thing. surely the cnn can detect that, right?
            # sample = torchvision.transforms.functional.invert(sample)
                # vflip(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


squish_to_linear = 128 * (32 // 2**3) * (256 // 2**3)
class OrientationCNN(Module):

    def __init__(self):
        super(OrientationCNN, self).__init__()
        # 3 = channel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(squish_to_linear, 128) # 128 * 28 * 28, 128) # 28 = 224 / 2 / 2 / 2
        # 65536 = 128 * (32 / 2**3) * (256 / 2**3) * 4 (batch)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) # oh good, we do a bit of pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # kernel_size = stride = 2
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, squish_to_linear) # 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return x

class PadToSize:
    def __init__(self, size, pad_with=0, center=False):
        self.size = size
        self.pad_with = pad_with
        self.center = center

    def __call__(self, img):
        # fill with white
        # if img is bigger than size, crop it
        # if img is smaller than size, pad it

        # if img.size[0] > self.size[0] or img.size[1] > self.size[1]:
        #     img = torchvision.transforms.functional.crop(img, 0, 0, self.size[0], self.size[1])
        # if self.center:
        vertmargin = max(self.size[0] - img.size[0], 0) / 2
        horizmargin = max(self.size[1] - img.size[1], 0) / 2
        return torchvision.transforms.functional.pad(img,
                (math.floor(vertmargin), math.floor(horizmargin), math.ceil(vertmargin), math.ceil(horizmargin)), self.pad_with)
        # return torchvision.transforms.functional.pad(img,
        #         (0, 0, self.size[0] - img.size[0], self.size[1] - img.size[1]), self.pad_with)
        # return PIL.ImageOps.expand(img, (0, 0, self.size[0] - img.size[0], self.size[1] - img.size[1]), self.pad_with)
        # none of these are working ;-; let's try numpy
    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size} pad={self.pad_with})'

# if running as main
if __name__ == '__main__':
    for fname in glob.iglob('../iamdataset/words-sample/**/*.*', recursive=True):
        print(fname)
    exit(0)
    dataset = OrientationImageFolder('./../iamdataset/words-sample')
    dataset.make_dataset('./../iamdataset/words-sample', {'normal': 0, 'flipped': 1})


# When creating a CNN for image classification, how many layers of each type (ie. convolutional, pooling, fully connected, etc.) is best? How deep should the layers be?


