import random
import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
FIXED_SEED = 0
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# ----------------- with Robust Inference in Degradation --------------------------
# 0.5 0.3 0.1
def add_gaussian_noise(image):
    np.random.seed(FIXED_SEED)
    std_choices = [0.5, 0.3, 0.1]
    std = np.random.choice(std_choices)
    noise = np.random.normal(0., std, image.shape)
    mask = np.random.binomial(1, 0.5, image.shape)  # Create a mask with 50% probability for each pixel
    image_with_noise = image + noise * mask  # Apply noise only where mask is 1
    return np.clip(image_with_noise, 0, 1)  # Make sure image is still in [0, 1]


# 0.01, 0.05, 0.1
def add_salt_and_pepper_noise(image):
    np.random.seed(FIXED_SEED)
    prob_choices = [0.01, 0.05, 0.1]
    salt_prob = np.random.choice(prob_choices)
    pepper_prob = np.random.choice(prob_choices)
    total_pixels = image.shape[1] * image.shape[2]

    # Create a mask with 50% probability for each pixel
    mask_salt = np.random.binomial(1, 0.5, (image.shape[1], image.shape[2]))
    mask_pepper = np.random.binomial(1, 0.5, (image.shape[1], image.shape[2]))

    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Generate coordinates where salt and pepper noise will be added
    salt_coords = np.column_stack(np.where(mask_salt == 1))
    pepper_coords = np.column_stack(np.where(mask_pepper == 1))

    # Randomly choose coordinates based on the probability
    chosen_salt_coords = salt_coords[np.random.choice(salt_coords.shape[0], num_salt, replace=False)]
    chosen_pepper_coords = pepper_coords[np.random.choice(pepper_coords.shape[0], num_pepper, replace=False)]

    # Add Salt noise
    image[:, chosen_salt_coords[:, 0], chosen_salt_coords[:, 1]] = 1

    # Add Pepper noise
    image[:, chosen_pepper_coords[:, 0], chosen_pepper_coords[:, 1]] = 0

    return image


# 1, 1.5, 3
def apply_gaussian_blur(image):
    np.random.seed(FIXED_SEED)
    sigma_choices = [1, 1.5, 3]
    sigma = np.random.choice(sigma_choices)
    return gaussian_filter(image, sigma=(0, sigma, sigma))  # 0 sigma for the channel axis


def random_degradation(image, image_index, degradation_prob=0.5):
    # 为这个特定图像设置随机种子
    random.seed(image_index)
    np.random.seed(image_index)

    degradation_types = [add_gaussian_noise, add_salt_and_pepper_noise, color_jitter, apply_gaussian_blur]
    if random.random() < degradation_prob:
        # 随机选择退化方法
        degradation = random.choice(degradation_types)
        # 应用退化并返回结果
        return degradation(image)
    else:
        # 如果没有退化，返回原始图像
        return image


def color_jitter(image):
    np.random.seed(FIXED_SEED)
    prob_choices = [0.3, 0.5, 0.7]
    brightness = np.random.choice(prob_choices)
    contrast = np.random.choice(prob_choices)
    saturation = np.random.choice(prob_choices)
    # Brightness
    brightness_factor = 1 + np.random.uniform(-brightness, brightness)
    image *= brightness_factor

    # Contrast
    contrast_factor = 1 + np.random.uniform(-contrast, contrast)
    image = (image - 0.5) * contrast_factor + 0.5

    # Saturation (across all channels)
    mean = np.mean(image, axis=0, keepdims=True)
    image = mean + (image - mean) * (1 + np.random.uniform(-saturation, saturation))

    return np.clip(image, 0, 1)  # Make sure image is still in [0, 1]


# ----------------- Data Augmentation --------------------
def data_augmentation_seg(image, label):
    mode = np.random.randint(0, 8)
    if mode == 0:
        # original
        return image, label
    elif mode == 1:
        # flip up and down
        return np.flip(image, axis=1), np.flip(label, axis=1)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image, axes=(1, 2)), np.rot90(label, axes=(1, 2))
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=(1, 2))
        label = np.rot90(label, axes=(1, 2))
        return np.flip(image, axis=1), np.flip(label, axis=1)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2, axes=(1, 2)), np.rot90(label, k=2, axes=(1, 2))
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=(1, 2))
        label = np.rot90(label, k=2, axes=(1, 2))
        return np.flip(image, axis=1), np.flip(label, axis=1)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3, axes=(1, 2)), np.rot90(label, k=3, axes=(1, 2))
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=(1, 2))
        label = np.rot90(label, k=3, axes=(1, 2))
        return np.flip(image, axis=1), np.flip(label, axis=1)


# ------------- Datasets Visualization ------------------
def visualize_augmented_samples(dataloader, num_samples=10, channels=[2, 3, 4]):
    """
    Visualize a few samples from the dataloader along with their corresponding ground truth labels.

    Parameters:
        dataloader (torch.utils.data.DataLoader): The dataloader.
        num_samples (int): Number of samples to visualize.
        channels (list of int): The channels to use for RGB visualization.
    """
    # Create a new figure
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 10))

    # Loop over a few batches to get num_samples samples
    sample_count = 0
    for i_batch, sample in enumerate(dataloader):
        original_images, augmented_images, degraded_images, labels = sample['original_image'], sample[
            'augmented_image'], \
            sample['degraded_image'], sample['degraded_label']

        for i in range(degraded_images.shape[0]):
            if sample_count >= num_samples:
                break

            # Create an RGB image: Use channels 2, 3, 4 as R, G, B
            def get_rgb_img(image):
                return image[channels, :, :].numpy().transpose((1, 2, 0))

            original_rgb = get_rgb_img(original_images[i])
            augmented_rgb = get_rgb_img(augmented_images[i])
            degraded_rgb = get_rgb_img(degraded_images[i])

            # Get the corresponding label
            label_img = labels[i, :, :].numpy().transpose((1, 2, 0))  # Assuming single-channel label

            # Display the image and label
            for ax, img, title in zip(axes[:, sample_count], [original_rgb, augmented_rgb, degraded_rgb, label_img],
                                      ["Original", "Augmented", "Degraded", "Ground Truth"]):
                if title == "Ground Truth":
                    ax.imshow(img[:, :, 1], cmap='gray')
                else:
                    ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"{title}")

            sample_count += 1

        if sample_count >= num_samples:
            break

    plt.show()


def visualize_augmented_ybsf(dataloader, num_samples=10, channels=[2, 3, 4]):
    # Create a new figure
    fig, axes = plt.subplots(4, num_samples, figsize=(20, 10))

    # Define the colormap for the labels
    label_mapping = {0: 'Background', 1: 'Clear Sky', 2: 'Sea Fog', 3: 'Cloud'}
    cmap = plt.cm.jet
    norm = plt.Normalize(0, 3)

    # Loop over a few batches to get num_samples samples
    sample_count = 0
    for i_batch, sample_dict in enumerate(dataloader):
        for i in range(sample_dict['degraded_image'].shape[0]):
            if sample_count >= num_samples:
                break

            # Create an RGB image: Use channels 2, 3, 4 as R, G, B
            def get_rgb_img(image):
                return image[channels, :, :].numpy().transpose((1, 2, 0))

            original_rgb = get_rgb_img(sample_dict['original_image'][i])
            augmented_rgb = get_rgb_img(sample_dict['augmented_image'][i])
            degraded_rgb = get_rgb_img(sample_dict['degraded_image'][i])

            # Get the corresponding label and convert to categorical label
            label_img = sample_dict['degraded_label'][i].argmax(dim=0).numpy()

            # Display the image and label
            for ax, img, title in zip(axes[:, sample_count], [original_rgb, augmented_rgb, degraded_rgb, label_img],
                                      ["Original", "Augmented", "Degraded", "Ground Truth"]):
                if title == "Ground Truth":
                    im = ax.imshow(img, cmap=cmap, norm=norm)
                else:
                    ax.imshow(img)
                ax.axis('off')
                ax.set_title(title)

            sample_count += 1

        if sample_count >= num_samples:
            break

    # Add a legend for the labels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=[0, 1, 2, 3], orientation='vertical', ax=axes.ravel().tolist(), pad=0.01)
    plt.show()


