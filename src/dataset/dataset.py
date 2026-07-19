import os
import numpy as np
from pathlib import Path
from torch.utils.data.dataset import Dataset
from PIL import Image


class LoveDaDataset(Dataset):
    dataset = Path(r"\data\raw\loveda\LoveDA")
# calculate the number of  examples in my train dataset
    def __init__(self,  dataset):
        self.dataset = dataset
        self.examples = list(self.dataset) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        # 1. get the image path
        image_path = self.examples[index]

        # 2. get the mask path
        mask_path = self.examples[index].parent / f"{self.examples[index].stem}_mask.png"

        # 3. download the image and convert it to RGB
        image = Image.open(image_path).convert("RGB")

        # 4. download the mask
        mask = Image.open(mask_path)

        # 5. Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        return image, mask