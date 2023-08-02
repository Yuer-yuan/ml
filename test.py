import os
import torch
from model import SRCNN
from config import *
from dataset import TestDataset
from torch.utils.data.dataloader import DataLoader
from utils import calc_psnr, load_ckpt
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def test():
    device = torch.device('cuda:0')
    model = SRCNN()
    model = model.to(device)

    test_dataset = TestDataset(test_file)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

    optimizer = torch.optim.SGD([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr * 0.1},
    ], lr=lr, momentum=0.9)

    start_epoch, best_epoch, best_psnr = load_ckpt(
        model, optimizer, from_pth=True)

    # validate
    model.eval()
    for i, data in enumerate(test_dataloader):
        input = data[0]
        label = data[1]
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(input).clamp(0., 1.)

        input = input[:, :, 6:-6, 6:-6]
        bicubic_psnr = calc_psnr(input, label)
        pred_psnr = calc_psnr(pred, label)

        input = input.mul(255.).cpu().squeeze().numpy()
        pred = pred.mul(255.).cpu().squeeze().numpy()
        label = label.mul(255.).cpu().squeeze().numpy()

        input = np.moveaxis(input, 0, -1).astype(np.uint8)
        pred = np.moveaxis(pred, 0, -1).astype(np.uint8)
        label = np.moveaxis(label, 0, -1).astype(np.uint8)

        plt.subplot(131), plt.imshow(label), plt.title('ORIGINAL')
        plt.subplot(132), plt.imshow(input), plt.title(
            "BICUBIC: {:.2f}".format(bicubic_psnr))
        plt.subplot(133), plt.imshow(
            pred), plt.title("SRCNN: {:.2f}".format(pred_psnr))
        plt.show()


if __name__ == '__main__':
    test()
