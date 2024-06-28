import os
from data_sfd.data_utils import data_augmentation_seg, random_degradation, visualize_augmented_samples, \
    visualize_augmented_ybsf
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torchvision.transforms import Resize
import torch.nn.functional as F

def one_hot(tensor, num_classes):
    return torch.nn.functional.one_hot(tensor, num_classes)


class YBSFDataset(Dataset):
    def __init__(self, root_dir, mode, imagesize=224, degrade_prob=0.5):
        self.label_mapping = {0: 'Background',
                              1: 'Clear Sky',
                              2: 'Sea Fog',
                              3: 'Cloud'}
        self.root_dir = root_dir
        self.image_folder = os.path.join(self.root_dir, 'images')
        self.label_folder = os.path.join(self.root_dir, 'labels')


        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.npy')]
        self.label_files = [f for f in os.listdir(self.label_folder) if f.endswith('.npy')]
        self.mode = mode
        self.imagesize = imagesize
        self.degrade_prob = degrade_prob

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        resize = Resize((self.imagesize, self.imagesize), antialias=True)
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])

        image = np.load(image_path)
        label = np.load(label_path)
        image, label = image.transpose((2, 0, 1)), np.expand_dims(label, axis=0)
        original_image, original_label = np.copy(image), np.copy(label)

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

        degraded_image_tensor = torch.as_tensor(degraded_image, dtype=torch.float32)
        label_tensor = torch.as_tensor(augmented_label, dtype=torch.int64)
        original_image_tensor = torch.as_tensor(original_image, dtype=torch.float32)
        original_label_tensor = torch.as_tensor(original_label, dtype=torch.int64)
        augmented_image_tensor = torch.as_tensor(augmented_image, dtype=torch.float32)
        original_label_tensor = one_hot(original_label_tensor.squeeze(), num_classes=4)
        label_tensor = one_hot(label_tensor.squeeze(), num_classes=4)
        original_label_tensor = original_label_tensor.permute(2, 0, 1)
        label_tensor = label_tensor.permute(2, 0, 1)
        degraded_image_tensor, label_tensor , original_image_tensor, original_label_tensor, augmented_image_tensor = resize(degraded_image_tensor), resize(label_tensor), resize(original_image_tensor), resize(original_label_tensor), resize(augmented_image_tensor)

        sample = {'degraded_image': degraded_image_tensor,
                  'degraded_label': label_tensor,
                  'original_image': original_image_tensor,
                  'original_label': original_label_tensor,
                  'augmented_image': augmented_image_tensor,
                  'augmented_label': label_tensor,
                  'id': idx
                  }
        return sample


def visualize_samples_with_labels(dataloader, num_samples=10, channels=[2,3,4]):
    # Define custom colormap
    custom_cmap = ListedColormap(['black', 'blue', 'yellow', 'green'])
    label_mapping = {0: 'Background', 1: 'Clear Sky', 2: 'Sea Fog', 3: 'Cloud'}

    fig, axes = plt.subplots(2, num_samples, figsize=(20, 10))

    # Create custom legend
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in
                      zip(custom_cmap.colors, label_mapping.values())]

    sample_count = 0
    for i_batch, (images, labels) in enumerate(dataloader):
        for i in range(images.shape[0]):
            if sample_count >= num_samples:
                break

            rgb_img = images[i, :, :, channels].numpy()
            label_img = labels[i, :, :].argmax(axis=-1).numpy()

            axes[0, sample_count].imshow(rgb_img)
            axes[0, sample_count].axis('off')
            axes[0, sample_count].set_title(f"Sample {sample_count + 1}")

            im = axes[1, sample_count].imshow(label_img, cmap=custom_cmap)
            axes[1, sample_count].axis('off')
            axes[1, sample_count].set_title("Ground Truth")

            sample_count += 1

        if sample_count >= num_samples:
            break

    # Add legend
    fig.legend(handles=legend_patches, loc='upper right')
    plt.show()


if __name__ == "__main__":
    dataset = YBSFDataset(root_dir='D:\\MS_Seg\\DATASET\\YBSF_SeaFog\\train', mode='train_degrade')
    ybsf_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    visualize_augmented_ybsf(ybsf_dataloader, num_samples=10, channels=[2, 3, 13])
