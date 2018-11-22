# coding:utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
import numpy as np

# import sys
# sys.path.insert(0,'C:/Users/50657/Desktop/faset_rcnn')
# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])


# 这里生成的anchors是对应到原图片大小的
# 返回的坐标形式：[x1,y1,x2,y2]
# 感觉base_size可以尝试一下32?
# 输出: 9个anchors
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=np.array([8, 16, 32])):
    """
    为什么不构造中心是(0,0)的anchors呢?   现在构造的anchors的中心是(7.5,7.5)
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # scales是array([ 8, 16, 32]), 指的就是之前的anchor_scales = [8, 16, 32]
    # 这是个1维向量,值为 [0, 0, 15, 15],这四个值分别代表[x1,y1,x2,y2]
    # 注意,这个anchor的边长是16,因为0也表示一个pixel
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    # 得到三个不同ratio的坐标,形如[x1,y1,x2,y2]
    # ratio_anchors: 3*4
    '''
    [[-3.5  2.  18.5 13. ]
     [ 0.   0.  15.  15. ]
     [ 2.5 -3.  12.5 18. ]]
    '''
    # 输出: anchor在三种长宽比的条件下生成的三个新anchors:x1,y1,x2,y2
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    # 垂直拼接
    # scales=[8,16,32]
    # anchors: 9*4
    # 每种长宽比下的anchor同时又对应三种不同的scale
    # 最后将这些anchors竖直排列起来
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(  ratio_anchors.shape[0])  ])
    return anchors

# _whctrs() 跟 _mkanchors()的功能正好是相反的
# 传入anchor左上角和右下角的坐标,返回anchor的w,h,x,y
# 输入: anchor:x1,y1,x2,y2
# 输出: x,y,w,h
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    # 返回anchor宽高以及anchor中心的坐标x,y
    # anchor:[0, 0, 15, 15],传入的anchor坐标有各种各样的
    w = anchor[2] - anchor[0] + 1 # 16
    h = anchor[3] - anchor[1] + 1 # 16
    x_ctr = anchor[0] + 0.5 * (w - 1) # 7
    y_ctr = anchor[1] + 0.5 * (h - 1) # 7
    return w, h, x_ctr, y_ctr

# _whctrs() 跟 _mkanchors()的功能正好是相反的
# 根据ws,hs以及中心坐标,计算anchor坐标(左上和右下角的坐标)
# 返回:anchors:[x1,y1,x2,y2]
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis] # [23,16,11]
    hs = hs[:, np.newaxis] # [12,16,6]
    # 水平拼接,保存的是[x1,y1,x2,y2],anchor左上和右下的坐标(难道是以base anchor的左上角为原点?)
    # anchors: 3*4
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

# 三种ratios
# 输入: anchor:x1,y1,x2,y2;长宽比
# 输出: anchor在三种长宽比的条件下生成的三个新anchors: x1,y1,x2,y2
def _ratio_enum(anchor, ratios):
    """
    为每种长宽比枚举出一组anchors
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    # anchor:[0, 0, 15, 15]
    # ratios=[0.5, 1, 2]
    # 输入:x1,y1,x2,y2 输出x,y,w,h
    w, h, x_ctr, y_ctr = _whctrs(anchor) 
    size = w * h # anchor面积 256
    # anchors中长宽1:2中最大为352x704，长宽2:1中最大736x384，基本是cover了800x600的各个尺度和形状
    size_ratios = size / ratios # anchor面积 512,256,128
    ws = np.round(np.sqrt(size_ratios)) # 求平方根再取整数,遵循四舍五入,[23,16,11]
    hs = np.round(ws * ratios)# [12,16,6]
    # 得到三个不同ratio的anchor坐标,形式如:[x1,y1,x2,y2]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# 每个anchor都是下面3*4 ndarray的每一行
    '''
    [[-3.5  2.  18.5 13. ]
     [ 0.   0.  15.  15. ]
     [ 2.5 -3.  12.5 18. ]]
    '''

# 三种scales
# 对于每一个anchor,都生成对应三种不同scale的anchor
# 输入:anchor的 x1,y1,x2,y2; scales:[ 8, 16, 32]
# 输出:anchors: x1,y1,x2,y2
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    # _scale_enum(ratio_anchors[i, :], scales)
    w, h, x_ctr, y_ctr = _whctrs(anchor) 
    ws = w * scales # 每个anchor对应三种不同的尺度
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print a
    print time.time() - t


    # print a.shape[0]
    for x in range(a.shape[0]):
        print _whctrs(a[x,:])
    # from IPython import embed; embed()
