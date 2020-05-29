# -*- coding: utf-8 -*-
# @Time    : 18-7-5 上午9:20
# @Author  : zhoujun
import os
import time

import cv2
# import matplotlib.pyplot as plt
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def unwarp(img, bm):
    w, h = img.shape[0], img.shape[1]
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0, :, :, :]
    bm0 = cv2.blur(bm[:, :, 0], (25, 25))
    bm1 = cv2.blur(bm[:, :, 1], (25, 25))
    bm0 = cv2.resize(bm0, (h, w))
    bm1 = cv2.resize(bm1, (h, w))
    bm = np.stack([bm0, bm1], axis=-1)
    bm = np.expand_dims(bm, 0)
    bm = torch.from_numpy(bm).double()

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()

    res = F.grid_sample(input=img, grid=bm)
    res = res[0].numpy().transpose((1, 2, 0))

    return res


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


class Pytorch_model:
    def __init__(self, model_path, net, img_h, img_w, img_channel=3, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.img_h = img_h
        self.img_w = img_w
        self.img_channel = img_channel
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            self.device = torch.device("cuda:0")
            self.net = torch.load(
                model_path, map_location=lambda storage, loc: storage.cuda(gpu_id))
        else:
            self.device = torch.device("cpu")
            self.net = torch.load(
                model_path, map_location=lambda storage, loc: storage.cpu())
        print('device:', self.device)

        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            net.load_state_dict(self.net['state_dict'])
            self.net = net
        self.net.eval()

    def predict(self, img):
        '''
        对传入的图像进行预测，支持图像地址
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        assert self.img_channel in [1, 3], 'img_channel must in [1.3]'

        if isinstance(img, str):  # read image
            assert os.path.exists(img), 'file is not exists'
            img = Image.open(img)
        if self.img_channel == 1 and img.mode == 'RGB':
            img = img.convert('L')
        w, h = img.size
        if w > h:
            ratio = h / w
            new_w = w // 16 * 16
            new_h = int(new_w * ratio)
        else:
            ratio = w / h
            new_h = h // 16 * 16
            new_w = int(new_h * ratio)

        img = img.resize((self.img_w, self.img_h))
        print(img.size)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        preds = self.net(tensor)
        # print(preds)
        # preds = preds[0].permute(1, 2, 0).detach().cpu().numpy()
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # unwarp_img = cv2.remap(img, preds[:, :, 0], preds[:, :, 1], cv2.INTER_LINEAR)
        # unwarp_img = cv2.resize(unwarp_img,(w,h))
        unwarp_img = unwarp(img, preds)
        return unwarp_img

    def predict_cv(self, img: np.ndarray or str):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img: 图像地址
        :param is_numpy:
        :return:
        '''
        # assert self.img_channel in [1, 3], 'img_channel must in [1.3]'

        # if isinstance(img, str):  # read image
        #     assert os.path.exists(img), 'file is not exists'
        #     img = cv2.imread(img, 0 if self.img_channel == 1 else 1)

        # if len(img.shape) == 2 and self.img_channel == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # elif len(img.shape) == 3 and self.img_channel == 1:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgorg = cv2.imread(img)
        imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(imgorg, (self.img_w, self.img_h))
        # img = img[:, :, ::-1]
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW
        img = np.expand_dims(img, 0)
        tensor = torch.from_numpy(img).float()
        h, w = img.shape[:2]
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # t_img = cv2.resize(img, (self.img_w, self.img_h))
        # print(t_img.shape)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        # tensor = transforms.ToTensor()(t_img)
        # tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        preds = self.net(tensor)
        # print(preds)
        # preds = preds[0].permute(1, 2, 0).detach().cpu().numpy()
        # unwarp_img = cv2.remap(t_img, preds[:, :, 0], preds[:, :, 1], cv2.INTER_LINEAR)
        # unwarp_img = cv2.resize(unwarp_img, (w, h))
        unwarp_img = unwarp(imgorg, preds)
        return unwarp_img, imgorg


if __name__ == '__main__':
    from models.deeplab_models.deeplab import DeepLab

    model_path = 'output/doc3d_data_deeplab/DocUnet_21_0.012450308240755322.pth'

    # model_path = './output/model.pkl'
    # img_path = '/data2/zj/data/add_bg_img_800_600/item1/0_0.jpg'
    # img_path = '/root/single'
    # 初始化网络
    net = DeepLab(backbone='resnet', output_stride=16, num_classes=2, pretrained=False)
    model = Pytorch_model(model_path, net=net, img_h=448, img_w=448, img_channel=3, gpu_id=2)
    for img_path in get_file_list('eval/input', p_postfix='.*'):
        if img_path.__contains__('result'):
            continue
        start = time.time()
        unwarp_img, img = model.predict_cv(img_path)
        print(time.time() - start)
        # 执行预测
        # 可视化
        save_path = os.path.splitext(img_path)[0].replace('input', 'unwarp')
        print(save_path)
        cv2.imwrite(save_path + '_epoch_21_result.jpg', unwarp_img[:, :, ::-1]*255)
        # plt.subplot(1, 2, 1)
        # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.title('input')
        # plt.imshow(img)
        # plt.subplot(1, 2, 2)
        # # unwarp_img = cv2.cvtColor(unwarp_img, cv2.COLOR_BGR2RGB)
        # plt.title('output')
        # plt.imshow(unwarp_img[:, :, ::-1]*255)
        # plt.savefig(save_path + '_plt_result.jpg', dpi=600)
        # # plt.show()
