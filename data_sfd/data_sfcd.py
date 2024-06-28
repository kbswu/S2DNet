"""
    Author: Wu Nan,
    Time: 9/30/2023
    Base Dataset and Generalized Functions for Sea Fog Dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from data_sfd.data_utils import data_augmentation_seg, random_degradation, visualize_augmented_samples
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class SeaFogDataset(Dataset):
    def __init__(self, root_dir, years, mode, degrade_prob=0.5, image_size=224):
        """
        Args:
            root_dir (str): Directory with all the years.
            years (list): List of years to include in the dataset.
        """
        self.root_dir = root_dir
        self.years = years
        self.mode = mode
        self.degrade_prob = degrade_prob
        self.imagesize = image_size

        # Collect all .npy files from the specified years
        self.file_list = []
        for year in self.years:
            image_folder = os.path.join(self.root_dir, year, 'image')
            self.file_list += [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if
                               fname.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # get image and label path
        resize = Resize((self.imagesize, self.imagesize), antialias=True)
        image_path = self.file_list[idx]
        year = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        label_path = os.path.join(self.root_dir, year, 'label', os.path.basename(image_path))

        # load image and label
        image = np.load(image_path) / 255.0
        label = np.round(np.load(label_path) / 255.0)

        # image: HWC -> CHW, label: HW -> 1HW
        image, label = image.transpose((2, 0, 1)), np.expand_dims(label, axis=0)
        original_image = np.copy(image)  # Deep copy
        original_label = np.copy(label)  # Deep copy

        # initialize some placeholders
        augmented_image = original_image.copy()
        augmented_label = original_label.copy()
        degraded_image = original_image.copy()

        # Apply data augmentation only for training modes
        if 'train' in self.mode:
            aug_image, aug_label = data_augmentation_seg(original_image, original_label)
            augmented_image, augmented_label = aug_image.copy(), aug_label.copy()
        if 'degrade' in self.mode:
            degraded_image = random_degradation(augmented_image.copy(), idx, degradation_prob=self.degrade_prob)

        # Convert to torch tensor
        degraded_image_tensor = torch.as_tensor(degraded_image, dtype=torch.float32)
        label_tensor = torch.as_tensor(augmented_label, dtype=torch.float32)
        original_image_tensor = torch.as_tensor(original_image, dtype=torch.float32)
        original_label_tensor = torch.as_tensor(original_label, dtype=torch.float32)
        augmented_image_tensor = torch.as_tensor(augmented_image, dtype=torch.float32)

        # calculate one-hot background
        background_label_tensor = 1 - label_tensor
        original_background_label_tensor = 1 - original_label_tensor
        label_tensor_bg = torch.cat([background_label_tensor, label_tensor], dim=0)
        original_label_tensor_bg = torch.cat([original_background_label_tensor, original_label_tensor], dim=0)
        degraded_image_tensor, label_tensor_bg, original_image_tensor_bg, original_label_tensor, augmented_image_tensor = resize(
            degraded_image_tensor), resize(label_tensor_bg), resize(original_image_tensor), resize(
            original_label_tensor_bg), resize(augmented_image_tensor)
        # futher process the label
        label_tensor_bg[label_tensor_bg >= 0.5] = 1
        label_tensor_bg[label_tensor_bg < 0.5] = 0

        original_label_tensor[original_label_tensor >= 0.5] = 1
        original_label_tensor[original_label_tensor < 0.5] = 0

        sample = {'degraded_image': degraded_image_tensor,
                  'degraded_label': label_tensor_bg,
                  'original_image': original_image_tensor_bg,
                  'original_label': original_label_tensor,
                  'augmented_image': augmented_image_tensor,
                  'augmented_label': label_tensor_bg,
                  'id': idx
                  }

        return sample


def build_dataloader(test_year):
    root_dir = r"D:/MS_Seg/DATASET/seafog_npy"
    years = ["2017", "2018", "2019", "2020"]
    train_years = [year for year in years if year != test_year]
    test_years = [test_year]
    train_dataset = SeaFogDataset(root_dir=root_dir, years=train_years)
    test_dataset = SeaFogDataset(root_dir=root_dir, years=test_years)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    # Dataset and DataLoader
    root_dir = "D:/MS_Seg/DATASET/seafog_npy"
    years = ["2017", "2018", "2019", "2020"]
    # Initialize the dataset and dataloader
    transformed_dataset = SeaFogDataset(root_dir=root_dir,
                                        years=years,
                                        mode='test_degrade'
                                        )
    viz_dataloader = DataLoader(transformed_dataset, batch_size=16, shuffle=True, num_workers=0)

    # Visualize some samples
    visualize_augmented_samples(viz_dataloader, num_samples=10, channels=[2, 3, 4])
