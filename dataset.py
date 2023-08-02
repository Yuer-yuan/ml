# methods to generate dataset from training and validation
# ref:
# https://zhuanlan.zhihu.com/p/431724297
# https://github.com/Sadisticheaven/SuperRestoration/blob/master/SRCNN/gen_datasets.py

# relationship between training set, validation set and testing set
# ref:
# https://blog.csdn.net/Chaolei3/article/details/79270939
import numpy as np
import h5py
from tqdm import tqdm
import os
from PIL import Image
from config import *
from torch.utils.data import Dataset


def gen_train():
    save_path = train_file
    h5_file = h5py.File(save_path, 'w')
    imgs_dir = os.path.join(datasets_dir, train_dataset)
    imgs = os.listdir(imgs_dir)
    padding = (in_sz - out_sz) // 2

    lr_subimgs = []
    hr_subimgs = []
    with tqdm(total=len(imgs)) as t:
        for img in imgs:
            img_path = os.path.join(imgs_dir, img)
            hr_img = Image.open(img_path)
            img_sz = (hr_img.width - hr_img.width %
                      scale, hr_img.height - hr_img.height % scale)
            hr_img = hr_img.crop((0, 0, img_sz[0], img_sz[1]))
            lr_img = hr_img.resize(
                (img_sz[0] // scale, img_sz[1] // scale), Image.BICUBIC)
            lr_img = lr_img.resize((img_sz[0], img_sz[1]), Image.BICUBIC)

            # change shape of [h, w, c] to [c, h, w] for training
            # ref: https://stackoverflow.com/questions/43829711/what-is-the-correct-way-to-change-image-channel-ordering-between-channels-first
            hr_img = np.array(hr_img).astype(np.float32)
            if (len(hr_img.shape) == 2):
                # gray img
                hr_img = np.expand_dims(hr_img, 0).repeat(3, axis=0)
            else:
                hr_img = np.moveaxis(hr_img, -1, 0)
            lr_img = np.array(lr_img).astype(np.float32)
            if (len(lr_img.shape) == 2):
                lr_img = np.expand_dims(lr_img, 0).repeat(3, axis=0)
            else:
                lr_img = np.moveaxis(lr_img, -1, 0)

            for r in range(0, img_sz[1] - in_sz + 1, stride):
                for c in range(0, img_sz[0] - in_sz + 1, stride):
                    lr_subimg = lr_img[:, r: r + in_sz, c: c + in_sz]
                    hr_subimg = hr_img[:, r + padding: r + padding +
                                       out_sz, c + padding: c + padding + out_sz]

                    lr_subimgs.append(lr_subimg)
                    hr_subimgs.append(hr_subimg)

            t.update(1)

    lr_subimgs = np.array(lr_subimgs)
    hr_subimgs = np.array(hr_subimgs)

    h5_file.create_dataset('data', data=lr_subimgs)
    h5_file.create_dataset('label', data=hr_subimgs)

    h5_file.close()


def gen_val():
    save_path = val_file
    h5_file = h5py.File(save_path, 'w')
    imgs_dir = os.path.join(datasets_dir, val_dataset)
    imgs = os.listdir(imgs_dir)
    padding = (in_sz - out_sz) // 2

    lr_group = h5_file.create_group('data')
    hr_group = h5_file.create_group('label')
    with tqdm(total=len(imgs)) as t:
        for i, img in enumerate(imgs):
            img_path = os.path.join(imgs_dir, img)
            hr_img = Image.open(img_path)
            img_sz = (hr_img.width - hr_img.width %
                      scale, hr_img.height - hr_img.height % scale)
            hr_img = hr_img.crop((0, 0, img_sz[0], img_sz[1]))
            lr_img = hr_img.resize(
                (img_sz[0] // scale, img_sz[1] // scale), Image.BICUBIC)
            lr_img = lr_img.resize((img_sz[0], img_sz[1]), Image.BICUBIC)

            hr_img = np.array(hr_img).astype(np.float32)
            if (len(hr_img.shape) == 2):
                hr_img = np.expand_dims(hr_img, 0).repeat(3, axis=0)
            else:
                hr_img = np.moveaxis(hr_img, -1, 0)
            lr_img = np.array(lr_img).astype(np.float32)
            if (len(lr_img.shape) == 2):
                lr_img = np.expand_dims(lr_img, 0).repeat(3, axis=0)
            else:
                lr_img = np.moveaxis(lr_img, -1, 0)

            lr_group.create_dataset(str(i), data=lr_img)
            hr_group.create_dataset(str(i), data=hr_img)
            t.update(1)

    h5_file.close()


def gen_test():
    save_path = test_file
    h5_file = h5py.File(save_path, 'w')
    imgs_dir = os.path.join(datasets_dir, test_dataset)
    imgs = os.listdir(imgs_dir)
    padding = (in_sz - out_sz) // 2

    lr_group = h5_file.create_group('data')
    hr_group = h5_file.create_group('label')
    with tqdm(total=len(imgs)) as t:
        for i, img in enumerate(imgs):
            img_path = os.path.join(imgs_dir, img)
            hr_img = Image.open(img_path)
            img_sz = (hr_img.width - hr_img.width %
                      scale, hr_img.height - hr_img.height % scale)
            hr_img = hr_img.crop((0, 0, img_sz[0], img_sz[1]))
            lr_img = hr_img.resize(
                (img_sz[0] // scale, img_sz[1] // scale), Image.BICUBIC)
            lr_img = lr_img.resize((img_sz[0], img_sz[1]), Image.BICUBIC)

            hr_img = np.array(hr_img).astype(np.float32)[
                padding: -padding, padding: -padding]
            if (len(hr_img.shape) == 2):
                hr_img = np.expand_dims(hr_img, 0).repeat(3, axis=0)
            else:
                hr_img = np.moveaxis(hr_img, -1, 0)
            lr_img = np.array(lr_img).astype(np.float32)
            if (len(lr_img.shape) == 2):
                lr_img = np.expand_dims(lr_img, 0).repeat(3, axis=0)
            else:
                lr_img = np.moveaxis(lr_img, -1, 0)

            lr_group.create_dataset(str(i), data=lr_img)
            hr_group.create_dataset(str(i), data=hr_img)
            t.update(1)

    h5_file.close()


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['data'][idx] / 255., f['label'][idx] / 255.

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class ValDataset(Dataset):
    def __init__(self, h5_file):
        super(ValDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['data'][str(idx)][:, :, :] / 255., f['label'][str(idx)][:, :, :] / 255.

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class TestDataset(Dataset):
    def __init__(self, h5_file):
        super(TestDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['data'][str(idx)][:, :, :] / 255., f['label'][str(idx)][:, :, :] / 255.

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


if __name__ == '__main__':
    gen_train()
    gen_val()
    gen_test()
