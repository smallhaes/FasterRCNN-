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
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import PIL


# 这个函数只是检查每一张图片？？
# 感觉这个函数被删减了
# 为imdb.roidb的每个元素(每个元素是字典)添加键:image,width,height,max_classes,max_overlaps
# 并没有生成anchor
def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    # 获取所有图片的size
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    roidb = imdb.roidb

    # 遍历每一张图片
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        # gt_overlaps是个什么样的arrary？？？？ 需要从imdb的roidb中寻找答案
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        
        # 为每个box 找个与它最match的gt box
        # 最大的IoU值 
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        # 最大的IoU值对应的gt box的索引，这里咋是类别了呐？
        max_classes = gt_overlaps.argmax(axis=1)


        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks 合理性检测；完整性检查
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

# 添加用来训练bbox回归器的信息
# 输入:roidb
# 输出:bbox target的均值,bbox target 的标准差                    bbox target就是各个偏移量
def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # 从gt_overlaps中的列的数量推断类别的数量
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        # roi和最match的gt box对应的最大IoU值
        max_overlaps = roidb[im_i]['max_overlaps']
        # 和roi最match的gt box对应的类别,相当于labels
        max_classes = roidb[im_i]['max_classes']
        # 输出的是targets.  第0列是box索引,后四列分别是x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
        roidb[im_i]['bbox_targets'] = \
                _compute_targets(rois, max_overlaps, max_classes)
    # 训练时是Ture
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Use fixed / precomputed "means" and "stds" instead of empirical values
        # BBOX_NORMALIZE_MEANS增加1个维度,变成2维,第0维重复21次,相当于有21行,第1维重复1次
        # BBOX_NORMALIZE_STDS增加1个维度,变成2维,第0维重复21次,相当于有21行,第1维重复1次
        means = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1))
        stds = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes, 1))
    else:
        # Compute values needed for means and stds
        # var(x) = E(x^2) - E(x)^2
        class_counts = np.zeros((num_classes, 1)) + cfg.EPS
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                if cls_inds.size > 0:
                    class_counts[cls] += cls_inds.size
                    sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                    squared_sums[cls, :] += \
                            (targets[cls_inds, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        # 平方的期望-期望的平方,结果再开根号
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print 'bbox target means:'
    print means
    print means[1:, :].mean(axis=0) # ignore bg class
    print 'bbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0) # ignore bg class

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        print "Normalizing targets"
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
    else:
        print "NOT normalizing targets"

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()
# 输入:rois,overlaps,labels
# 输出: targets:第0列是box索引,后四列分别是x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
# _compute_targets(rois, max_overlaps, max_classes)
def _compute_targets(rois, overlaps, labels):
    # 计算某张图片的回归偏移量
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return np.zeros((rois.shape[0], 5), dtype=np.float32)
    # Indices of examples for which we try to make predictions
    # 这样也会把gt_inds取出来啊!???
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    # 对于每个ex roi ,与它IoU最大的gt box的索引
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    # box的x1,y1,x2,y2
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    # 第0列是box的索引
    targets[ex_inds, 0] = labels[ex_inds]
    # 第1-4列是box的偏移量:x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets
