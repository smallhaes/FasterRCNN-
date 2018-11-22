# coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2


# 输入:图片的列表
# 输出:图片信息的blob形式:索引,高,宽,通道
def im_list_to_blob(ims):
    # 将图片们转换成网络的输入形式(blob)
    """Convert a list of images into a network input.
    假设图片们已经去均值,调成BGR顺序等等(已经对图片们进行了预处理)
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    # 为每一张图片构造这种blob,作为网络的输入
    for i in xrange(num_images):
        im = ims[i]
        # blob的第0维是图片索引,第1维是图片的高度,第2维是图片的宽度,第3维是图片的通道
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


# 输入:图片;像素均值;目标尺寸600;最大尺寸1000
# 输出: 去均值且放缩到输入网络的图片大小;放缩倍数
# 这个函数处理过的图片尺寸是网络输入的尺寸
def prep_im_for_blob(im, pixel_means, target_size, max_size):
    # 对图片进行去均值处理并进行尺度上的变换
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    # 去均值
    im -= pixel_means
    im_shape = im.shape
    # 最短边的值
    im_size_min = np.min(im_shape[0:2])
    # 最长边的值
    im_size_max = np.max(im_shape[0:2])
    # 目标尺寸比上最短边的值,相当于放缩的倍数
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    # 使用上面的放缩倍数对最长边进行放大,如果结果大于max_size则需要缩小放缩倍数
    if np.round(im_scale * im_size_max) > max_size:
        # 进入这个判断语句,说明放缩倍数过大,需要缩小
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
