# -*- coding: utf-8 -*-
# @Time    : 18-7-3 上午10:06
# @Author  : zhoujun
import os
import pathlib

import tqdm


def check():
    img_path = '/root/doc_unet_dataset/data_gen/'
    img_path = pathlib.Path(img_path)

    img_list = list(img_path.rglob('*.jpg'))
    print('start')
    pbar = tqdm.tqdm(total=len(img_list))
    for img in img_list:
        npy_path = pathlib.Path(str(img) + '.npy')
        if not img.exists() or img.stat().st_size <= 0 or not npy_path.exists() or npy_path.stat().st_size <= 0:
            if img.exists():
                os.remove(str(img))
            if npy_path.exists():
                os.remove(str(npy_path))
            print('img or npy is bad:{0},{1}'.format(img, npy_path))
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    check()
