import os
from typing import Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset


class MaterialDataset(Dataset):
    def __init__(self, root_dir: str, materials: List[str], split: str,
                 class_to_idx: Dict[str, int], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = class_to_idx
        self.classes = sorted(self.class_to_idx.keys())

        mats = sorted(materials)
        for material in mats:
            material_path = os.path.join(self.root_dir, material, split)
            if not os.path.isdir(material_path):
                print(f"Warning: Directory not found for {material} in split {split}")
                continue

            for class_name in sorted(os.listdir(material_path)):
                if class_name in self.class_to_idx:
                    class_path = os.path.join(material_path, class_name)
                    if os.path.isdir(class_path):
                        for img_name in sorted(os.listdir(class_path)):
                            img_path = os.path.join(class_path, img_name)
                            self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
