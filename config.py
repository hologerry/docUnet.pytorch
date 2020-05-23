# -*- coding: utf-8 -*-
# @Time    : 18-11-27 下午1:11
# @Author  : zhoujun

trainroot = '/root/doc_unet_dataset/data_gen/'
output_dir = 'output/docunet_add_bg_img_800_600_item_origin_docunet'

gpu_id = 0
workers = 6
start_epoch = 0
epochs = 500

train_batch_size = 3
back_step = 10

lr = 1e-4
end_lr = 1e-7
lr_decay = 0.1
lr_decay_step = 50
display_interval = 100
restart_training = True
# checkpoint = 'output/deeplab_add_bg_img_800_600_item_origin_deeplab_drn/DocUnet_100_39.09240816952138.pth'
checkpoint = ''

# random seed
seed = 2
