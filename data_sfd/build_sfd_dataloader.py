import torch
from torch.utils.data import DataLoader
import os

from data_sfd.data_sfcd import SeaFogDataset
from data_sfd.data_ybsf import YBSFDataset


def build_seafog_dataset(train_mode, test_mode, batch_size, image_size):
    root_dir = r"D:/MS_Seg/DATASET/seafog_npy"
    years = ["2017", "2018", "2019", "2020"]
    train_years = [year for year in years if year not in ["2019", "2020"]]
    val_years = ["2019"]
    test_years = ["2020"]
    train_dataset = SeaFogDataset(root_dir=root_dir, years=train_years, mode=train_mode, image_size=image_size, degrade_prob=0.5)
    val_dataset = SeaFogDataset(root_dir=root_dir, years=val_years, mode=test_mode, image_size=image_size, degrade_prob=1)
    test_dataset = SeaFogDataset(root_dir=root_dir, years=test_years, mode=test_mode, image_size=image_size, degrade_prob=1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_dataloader, val_dataloader, test_dataloader


def build_ybsf_dataset(train_mode, test_mode, batch_size, image_size):
    train_root_dir = r"D:/MS_Seg/DATASET/YBSF_SeaFog/train"
    test_root_dir = r"D:/MS_Seg/DATASET/YBSF_SeaFog/test"
    val_root_dir = r"D:/MS_Seg/DATASET/YBSF_SeaFog/val"
    train_dataset = YBSFDataset(root_dir=train_root_dir, mode=train_mode, imagesize=image_size, degrade_prob=0.5)
    test_dataset = YBSFDataset(root_dir=test_root_dir, mode=test_mode, imagesize=image_size, degrade_prob=1)
    val_dataset = YBSFDataset(root_dir=val_root_dir, mode=test_mode, imagesize=image_size, degrade_prob=1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_dataloader, val_dataloader, test_dataloader