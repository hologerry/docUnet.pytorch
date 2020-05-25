# -*- coding: utf-8 -*-
# @Time    : 2018/6/13 15:01
# @Author  : zhoujun
import collections
import os
import pathlib

import cv2
import hdf5storage as h5
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as Data
from natsort import natsorted
from PIL import Image
from torchvision import transforms


def get_file_list(folder_path: str, p_postfix: str or list = ['.jpg'], sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = []
    if sub_dir:
        for rootdir, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(rootdir, file)
                for p in p_postfix:
                    if os.path.isfile(file_path) and (file_path.endswith(p) or p == '.*'):
                        file_list.append(file_path)
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            for p in p_postfix:
                if os.path.isfile(file_path) and (file_path.endswith(p) or p == '.*'):
                    file_list.append(file_path)
    return natsorted(file_list)


class MyDataSet(Data.Dataset):
    def __init__(self, txt, data_shape, channel=3, transform=None, target_transform=None):
        '''
        :param txt: 存放图片和标签的文本，其中数据和标签以空格分隔，一行代表一个样本
        :param data_shape: 图片的输入大小
        :param channel: 图片的通道数
        :param transform: 数据的tarnsform
        :param target_transform: 标签的target_transform
        '''
        with open(txt, 'r') as f:
            data = list(line.strip().split(' ') for line in f if line)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.data_shape = data_shape
        self.channel = channel

    def __readimg__(self, img_path, transform):
        img = cv2.imread(img_path, 0 if self.channel == 1 else 3)
        img = cv2.resize(img, (self.data_shape[0], self.data_shape[1]))
        img = np.reshape(
            img, (self.data_shape[0], self.data_shape[1], self.channel))
        if transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        img_path, label_path = self.data[index]
        return self.__readimg__(img_path, self.transform), self.__readimg__(img_path, self.target_transform)

    def __len__(self):
        return len(self.data)


class ImageData(Data.Dataset):
    def __init__(self, img_root, transform=None, t_transform=None):
        self.image_path = get_file_list(
            img_root, p_postfix=['.jpg'], sub_dir=True)
        self.image_path = [
            x for x in self.image_path if pathlib.Path(x).stat().st_size > 0]

        self.label_path = [x + '.npy' for x in self.image_path]
        # self.label_path = [x.replace('add_bg_img_2000_1180_nnn', 'src_img') for x in self.image_path]
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, index):
        # image = Image.open(self.image_path[index])
        # label = Image.open(self.label_path[index])
        # label = label.resize(image.size)
        image = cv2.imread(self.image_path[index])
        label = np.load(self.label_path[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)

class Doc3dDataset(Data.Dataset):
    """
    Dataset for RGB Image -> Backward Mapping.
    """

    def __init__(self, root, split='train', is_transform=True,
                 img_size=448, augmentations=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 2  # target number of channel
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        for split in ['train', 'val']:
            path = os.path.join(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # 1/824_8-cp_Page_0503-7Nw0001
        im_name = self.files[self.split][index]

        im_path = os.path.join(self.root, 'img', im_name + '.png')
        im = Image.open(im_path).convert('RGB')

        bm_path = os.path.join(self.root, 'bm', im_name + '.mat')
        bm = h5.loadmat(bm_path)['bm']

        if self.is_transform:
            image, label = self.transform(im, bm)
        return image, label

    def transform(self, img, bm, chk=None):
        img = img.resize(self.img_size)

        bm = bm.astype(float)
        bm = bm / np.array([448.0, 448.0])
        bm = (bm - 0.5) * 2
        bm0 = cv2.resize(bm[:, :, 0], (self.img_size[0], self.img_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.img_size[0], self.img_size[1]))

        label = np.stack([bm0, bm1], axis=0)

        # to torch
        image = transforms.ToTensor()(img)
        label = torch.from_numpy(label).float()  # NCHW

        return image, label


if __name__ == '__main__':
    img_path = 'data/add_bg_img_2000_1180_nnn'
    test_data = ImageData(img_path, transform=transforms.ToTensor(), t_transform=transforms.ToTensor())
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=3)
    loss_fn = torch.nn.MSELoss()
    for img, label in test_loader:
        #     print(img[0].permute(1,2,0).numpy().shape)
        #     print(label.shape)
        #     print(img.dtype)
        #     print(img.shape)
        loss = loss_fn(img, label)
        show_img = img[0].permute(1, 2, 0).numpy()
        plt.imshow(show_img)
        plt.show()
        label = label[0].permute(1, 2, 0).numpy()
        label_img = cv2.remap(
            show_img, label[:, :, 0], label[:, :, 1], cv2.INTER_LINEAR)
        plt.imshow(label_img)
        plt.show()
        break
