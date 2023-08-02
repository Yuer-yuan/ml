import torch
import numpy as np
import csv
from config import *


def calc_psnr(img1, img2):
    if isinstance(img1, torch.Tensor):
        psnr = 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
    else:
        psnr = 10. * np.log10(1. / np.mean((img1 - img2) ** 2))
    return psnr.item()


class Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def load_ckpt(model, optimizer, from_pth=False):
    if not from_pth:
        with open(csv_file, 'w', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'psnr'])
        start_epoch = 0
        best_epoch = 0
        best_psnr = 0.

    else:
        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt['model'])
        model.eval()
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_epoch = ckpt['best_epoch']
        best_psnr = ckpt['best_psnr']
        print(
            f"start from epoch {start_epoch}, loss {ckpt['loss']}, psnr {ckpt['psnr']}")

    return start_epoch, best_epoch, best_psnr


def save_ckpt(model, optimizer, epoch, loss, psnr, best_epoch, best_psnr):
    if psnr > best_psnr:
        best_epoch = epoch
        best_psnr = psnr

    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'psnr': psnr,
        'best_epoch': best_epoch,
        'best_psnr': best_psnr,
    }

    torch.save(state_dict, ckpt_file)
    with open(csv_file, 'a', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss, psnr])

    return best_epoch, best_psnr
