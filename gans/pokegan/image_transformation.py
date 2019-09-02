import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from src import *


def get_dataset(transformation=None):
    return dset.ImageFolder(POKEMON_DATA_DIR, transform=transformation)


def get_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)


def show_image_sample():
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NUMBER_GPUS > 0) else "cpu")
    dataset = get_dataset(transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    dataloader = get_dataloader(dataset)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    show_image_sample()
