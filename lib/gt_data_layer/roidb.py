# coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# 通过添加一些metadata将roidb变成可训练的
"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
from utils.boxes_grid import get_boxes_grid
import scipy.sparse
import PIL
import math
import os
import cPickle
import pdb

# 丰富imdb中的roidb，通过增加可求导变量的方法。这样有助于训练，或者说更能帮助训练rpn网络
# 输入：imdb
# 输出：为imdb中的roidb的每个元素(每个元素是字典)添加键'info_boxes'，对应的value是个二维数组，共18列，由(cx, cy, scale_ind, box, scale_ind_map, box_map, gt_label, gt_sublabel, target)这些信息组成；将扩充后的imdb'roidb写入文件

# enrich过程：对每一张图片都进行了后面的操作：生成feamap所有点对应的所有anchors，为每个anchor找到最match的b box，再从这里面选择符合fg要求的anchor， 计算fg对应的anchor和gt box的偏移量， 将anchor的中心坐标，长宽以及对应到原图片上的长宽添加到每一张图片的info_boxes中(roidb[i]['info_boxes']),处理完所有图片后将roidb写入文件
def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities(可以求导的量) that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    # 如果有cache文件，加载后直接返回即可
    cache_file = os.path.join(imdb.cache_path, imdb.name + '_gt_roidb_prepared.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            imdb._roidb = cPickle.load(fid)
        print '{} gt roidb prepared loaded from {}'.format(imdb.name, cache_file)
        return

    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        # 这应该是gt box
        # roidb中的box并没有对应到原图！！！！！！！！！！！！！！！！！！
        boxes = roidb[i]['boxes']
        labels = roidb[i]['gt_classes']
        # feamap每个点9个box，每个box对应两个概率：是fg的概率；不是bg的概率
        # 生成的就是个空array
        # array([], shape=(0, 18), dtype=float32)
        info_boxes = np.zeros((0, 18), dtype=np.float32)

        if boxes.shape[0] == 0:
            roidb[i]['info_boxes'] = info_boxes
            continue

        # compute grid boxes
        s = PIL.Image.open(imdb.image_path_at(i)).size
        image_height = s[1]
        image_width = s[0]
        # 输入：图片的真是高度和宽度
        # 输出：boxes_grid：非常多(feamap所有点的数量*num_aspect)个[x1,y1,x2,y2], centers[:,0], centers[:,1]
        # 输出：box在原图中的左上角和右下角坐标；feature map中各个点对应的x坐标和y坐标
        # 这个box不是gt，这里是给feature map中的每个点生成多个box(不同比例的)    roidb中的box是 gt
        boxes_grid, cx, cy = get_boxes_grid(image_height, image_width)
        

        # Scales to use during training (can list multiple scales)
        # Each scale is the pixel size of an image's shortest side
        #__C.TRAIN.SCALES = (600,)
        # for each scale
        for scale_ind, scale in enumerate(cfg.TRAIN.SCALES):
            # scale应该是16
            boxes_rescaled = boxes * scale

            # compute overlap
            overlaps = bbox_overlaps(boxes_grid.astype(np.float), boxes_rescaled.astype(np.float))
            # 为每个box 找个与它最match的gt box
            # 最大的IoU值
            max_overlaps = overlaps.max(axis = 1)
            # 最大的IoU值对应的gt box的索引
            argmax_overlaps = overlaps.argmax(axis = 1)
            # 最match的gt box对应的类别
            max_classes = labels[argmax_overlaps]

            # select positive boxes
            fg_inds = []
            # 遍历所有类别，找出满足条件的boxes作为fg
            for k in xrange(1, imdb.num_classes):
                # IoU超过一定阈值的box才是fg！
                fg_inds.extend(np.where((max_classes == k) & (max_overlaps >= cfg.TRAIN.FG_THRESH))[0])

            if len(fg_inds) > 0:
                # fg对应的gt box的索引
                gt_inds = argmax_overlaps[fg_inds]
                # bounding box regression targets
                # 计算当前fg box 和其对应的 gt box 的偏移量
                # 返回值是2维的，有4列。第0列：x的偏移量；第1列：y的偏移量；第2列：w的伸缩量；第4列：h的伸缩量
                gt_targets = _compute_targets(boxes_grid[fg_inds,:], boxes_rescaled[gt_inds,:])
                
                # scale mapping for RoI pooling
                # cfg中没有这个变量？？？
                scale_ind_map = cfg.TRAIN.SCALE_MAPPING[scale_ind]
                scale_map = cfg.TRAIN.SCALES[scale_ind_map]

                # 创建fg对应的list
                # contruct the list of positive boxes
                # (cx, cy, scale_ind, box, scale_ind_map, box_map, gt_label, gt_sublabel, target)
                # 这里的18可不是9个anchor，而是1个anchor，用了18列存储相关信息
                info_box = np.zeros((len(fg_inds), 18), dtype=np.float32)
                info_box[:, 0] = cx[fg_inds]
                info_box[:, 1] = cy[fg_inds]
                info_box[:, 2] = scale_ind
                info_box[:, 3:7] = boxes_grid[fg_inds,:]
                info_box[:, 7] = scale_ind_map
                info_box[:, 8:12] = boxes_grid[fg_inds,:] * scale_map / scale
                info_box[:, 12] = labels[gt_inds]
                info_box[:, 14:] = gt_targets
                info_boxes = np.vstack((info_boxes, info_box))

        roidb[i]['info_boxes'] = info_boxes

    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote gt roidb prepared to {}'.format(cache_file)

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'info_boxes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]

    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 4))
    squared_sums = np.zeros((num_classes, 4))
    for im_i in xrange(num_images):
        targets = roidb[im_i]['info_boxes']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 12] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 14:].sum(axis=0)
                squared_sums[cls, :] += (targets[cls_inds, 14:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # Normalize targets
    for im_i in xrange(num_images):
        targets = roidb[im_i]['info_boxes']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 12] == cls)[0]
            roidb[im_i]['info_boxes'][cls_inds, 14:] -= means[cls, :]
            if stds[cls, 0] != 0:
                roidb[im_i]['info_boxes'][cls_inds, 14:] /= stds[cls, :]

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()




# 输入： anchors/proposals 和 gt boxes
# 输出：是2维的，有4列。第0列：x的偏移量；第1列：y的偏移量；第2列：w的伸缩量；第4列：h的伸缩量
def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image. The targets are scale invariance（目标值是尺度不变的）"""

    # 对于anchors/proposals
    # x2-x1
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
    # y2-y1
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
    # 中心坐标x
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    # 中心坐标y
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # 对于gt boxes
    # x2-x1
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
    # y2-y1
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
    # 中心坐标x
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    # 中心坐标y
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.zeros((ex_rois.shape[0], 4), dtype=np.float32)
    targets[:, 0] = targets_dx
    targets[:, 1] = targets_dy
    targets[:, 2] = targets_dw
    targets[:, 3] = targets_dh
    return targets
