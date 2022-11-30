from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class FashionDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()

        self.path = Path(path)
        images = [p.name for p in (self.path / "images").glob("*.jpg")]

        people_photos = defaultdict(list)
        for file in images:
            p_name = file.split("_")[2]
            people_photos[p_name].append(file)

        self.files = list(filter(lambda x: len(x) > 1, people_photos.values()))
        self.cache = dict()

    def __len__(self):
        return len(self.files)

    def load_file(self, file: Path):
        file = str(file)
        if file in self.cache:
            return self.cache[file]

        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)) / 255).float()
        self.cache[file] = image
        return image

    def __getitem__(self, idx: int):
        a, b = np.random.choice(len(self.files[idx]), 2, False)
        a = self.files[idx][a]
        b = self.files[idx][b]

        s_real = self.load_file(self.path / "images" / a)
        s_pose = self.load_file(self.path / "seg_maps" / a)
        s_app = self.load_file(self.path / "texture_maps" / a)
        t_real = self.load_file(self.path / "images" / b)
        t_pose = self.load_file(self.path / "seg_maps" / b)

        return s_real, s_pose, s_app, t_real, t_pose
