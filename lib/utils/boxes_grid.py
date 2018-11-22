#coding:utf-8
# --------------------------------------------------------
# Subcategory CNN
# Copyright (c) 2015 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
import math
from fast_rcnn.config import cfg


# 对应论文中的话，这个函数就是为feature map中的每个点生成9个anchor boxes用的！！
# 输入：图片的真是高度和宽度
# 输出：boxes_grid：非常多(feamap所有点的数量*num_aspect)个[x1,y1,x2,y2], centers[:,0], centers[:,1]
# 输出：box在原图中的左上角和右下角坐标；feature map中各个点对应的x坐标和y坐标
def get_boxes_grid(image_height, image_width):
    """
    在图像网格上返回boxes
    构建坐标系：先在box构建坐标系，再放大到原图中去！！！！
    Return the boxes on image grid.
    """

    # height and width of the heatmap
    if cfg.NET_NAME == 'CaffeNet':
        height = np.floor((image_height * max(cfg.TRAIN.SCALES) - 1) / 4.0 + 1)
        height = np.floor((height - 1) / 2.0 + 1 + 0.5)
        height = np.floor((height - 1) / 2.0 + 1 + 0.5)

        width = np.floor((image_width * max(cfg.TRAIN.SCALES) - 1) / 4.0 + 1)
        width = np.floor((width - 1) / 2.0 + 1 + 0.5)
        width = np.floor((width - 1) / 2.0 + 1 + 0.5)
    elif cfg.NET_NAME == 'VGGnet':
        # Scales to use during training (can list multiple scales)
        # Each scale is the pixel size of an image's shortest side
        #__C.TRAIN.SCALES = (600,)
        # sacale表示图片最短边的像素尺度，比如600
        #  +0.5 是为了实现四舍五入(1.6+0.5   1.2+0.5)，比较巧妙的一种方法，为什么不用np.round()?
        # 搞不懂为什么image_height * max(cfg.TRAIN.SCALES)？？？？？？？？  TRAIN.SCALES应该是1
        height = np.floor(image_height * max(cfg.TRAIN.SCALES) / 2.0 + 0.5)
        height = np.floor(height / 2.0 + 0.5)
        height = np.floor(height / 2.0 + 0.5)
        height = np.floor(height / 2.0 + 0.5)

        width = np.floor(image_width * max(cfg.TRAIN.SCALES) / 2.0 + 0.5)
        width = np.floor(width / 2.0 + 0.5)
        width = np.floor(width / 2.0 + 0.5)
        width = np.floor(width / 2.0 + 0.5)
    else:
        assert (1), 'The network architecture is not supported in utils.get_boxes_grid!'

    # compute the grid box centers
    # 这里应该是feature map的h和w， 确实是！
    # 注意这种生成坐标的方式，而不是想着用双层循环！
    h = np.arange(height)
    w = np.arange(width)
    y, x = np.meshgrid(h, w, indexing='ij') 
    centers = np.dstack((x, y))
    # centers包含两列，第0列表示x坐标，第1列表示y坐标
    centers = np.reshape(centers, (-1, 2))
    num = centers.shape[0]

    # compute width and height of grid box
    # box的面积
    area = cfg.TRAIN.KERNEL_SIZE * cfg.TRAIN.KERNEL_SIZE
    aspect = cfg.TRAIN.ASPECTS  # height / width
    num_aspect = len(aspect) # eg：4种长宽比
    # width and height of grid box
    # widths：1*4
    widths = np.zeros((1, num_aspect), dtype=np.float32)
    # heights：1*4
    heights = np.zeros((1, num_aspect), dtype=np.float32)
    for i in xrange(num_aspect):
        widths[0,i] = math.sqrt(area / aspect[i])
        heights[0,i] = widths[0,i] * aspect[i]

    # construct grid boxes
    # 每个元素在第0维上重复4次，相当于feamap中每个点对应4个box，原论文生成的是9个
    centers = np.repeat(centers, num_aspect, axis=0)
    # num先补充到长度为2，即：(1,1000多), 接着，widths在第0维上重复1次，在第1维上重复1000多次，也就是feamap每个点对应4种宽度
    # num应该是feature map中的点的个数，相当于每个点都有num_aspect个grid box
    # 转置后widths:4000多，1； 对于feature map中的所有点来说，共有4000多个grid box
    widths = np.tile(widths, num).transpose()
    # 转置后heights:4000多，1
    heights = np.tile(heights, num).transpose()

    # 下面四句其实就能说明，坐标系的原点在左上方或者左下方都可以
    # 甚至原点在右上方或者右下方也行
    # 我就把原点选在左上方吧
    # x1：grid box左端坐标       x2：grid box右端坐标
    x1 = np.reshape(centers[:,0], (-1, 1)) - widths * 0.5
    x2 = np.reshape(centers[:,0], (-1, 1)) + widths * 0.5
    # y1：grid box上端坐标       y2：grid box下端坐标
    y1 = np.reshape(centers[:,1], (-1, 1)) - heights * 0.5
    y2 = np.reshape(centers[:,1], (-1, 1)) + heights * 0.5
    
    # SPATIAL_SCALE = 1/16 = 0.0625
    # 一定要注意，这里的boxes_grid对应的是原图片中的坐标！ （放大了16倍）
    # grid中心坐标仍是feature map中每个点的坐标
    boxes_grid = np.hstack((x1, y1, x2, y2)) / cfg.TRAIN.SPATIAL_SCALE

    return boxes_grid, centers[:,0], centers[:,1]
