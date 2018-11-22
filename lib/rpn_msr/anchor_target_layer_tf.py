# coding:utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
# 
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
# 获得尺度无关的偏移量和伸缩量
from fast_rcnn.bbox_transform import bbox_transform
import pdb

DEBUG = False


# 怎么保证为feature map的每个pixel生成gt anchors, axis=1 !!!!!!!666  拿炉火刀

# 这个文件就是对h*w这么大小的feature map,进行操作,每个点生成A个anchors

# 这个文件是关于RPN的??? proposal_target和 proposal_layer是关于 ROI的?????
# 这个文件的标签有三种:1,0,-1

# 输入:rpn_cls_score(只用来传递feature map的shape, 用'rpn_bbox_pred'也行,因为不涉及通道)
# 输出:rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights
# 输出:标签rpn_labels(shape很奇怪:(1, 1, A * height, width)),偏移量rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights
# 最可怕的是输出中的rpn_labels和rpn_bbox_targets在训练时是作为gt使用的!!!!! 这个文件到底是何方神圣??? 不是有gt_boxes吗???
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):
    """
    将anchors分配给gt targets
    生成anchor分类标签 (标签信息在gt_boxes的第axis=4列)
    生成bounding-box 回归targets (也就是anchors相对gt的偏移量tx,ty,tw,th)
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    在proposal_target_layer中是：
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6))
    # 实际训练时,anchor_scales有3个元素,也就是对应3中不同的scale
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0] # 9

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # 允许少量的box在边缘上：坐标轴上或者略超过坐标轴
    # allow boxes to sit over the edge by a small amount
    _allowed_border =  0

    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]
    # im_info[0]中有图片的H，W，scale=16(缩放倍数)
    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    # feature map的height和width
    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes

    # 接下来的meshgrid非常巧妙
    #  之前生成了9种基本的anchors,现在要对这些anchors进行各种平移，使其布满整张图片
    # 1. Generate proposals from bbox deltas and shifted anchors
    # 放缩到原图大小
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    # 用来构造所有组合的二维数组
    # shift_x 和 shift_y各有 width*height 个元素
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 为什么构造了两份一样的二维数组？而且还要transpose.  因为图片的坐标格式是x1,y1,x2,y2. 所以这样做能够保证x1,x2平移一样的量.对y1,y2也是如此
    # (width*height,4)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0] # K=width*height
    # 通过np相加,使得feature map的每一点有9种anchors,这9种anchors是对应到原图大小的
    # 容易想象很多anchors都超过边界了,得配合_allowed_border剔除一些过分越界的anchors
    # K,A,4
    '''
    最初的9个anchors的w,h,x,y数据, 
    (184.0, 96.0, 7.5, 7.5)
    (368.0, 192.0, 7.5, 7.5)
    (736.0, 384.0, 7.5, 7.5)
    (128.0, 128.0, 7.5, 7.5)
    (256.0, 256.0, 7.5, 7.5)
    (512.0, 512.0, 7.5, 7.5)
    (88.0, 176.0, 7.5, 7.5)
    (176.0, 352.0, 7.5, 7.5)
    (352.0, 704.0, 7.5, 7.5)

    用的时候是x1,y1,x2,y2形式
    [[ -84.  -40.   99.   55.]
     [-176.  -88.  191.  103.]
     [-360. -184.  375.  199.]
     [ -56.  -56.   71.   71.]
     [-120. -120.  135.  135.]
     [-248. -248.  263.  263.]
     [ -36.  -80.   51.   95.]
     [ -80. -168.   95.  183.]
     [-168. -344.  183.  359.]]
    '''
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    # 这里还没有筛选anchors
    all_anchors = all_anchors.reshape((K * A, 4))
    # shift后所有的anchor数量=  width*height*9
    total_anchors = int(K * A)

    # only keep anchors inside the image
    # 在all_anchors的基础上筛选合适的索引,所以inds_inside和rpn_cls_prob_reshape的索引是一一对应的
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors.shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    # 将所有值初始化为-1
    labels.fill(-1)



    # 在poposal_layer_tf.py中会使anchors加上bbox_deltas(微调) 然后再在poposal_target_layer_tf.py使用overlaps
    # 在这个anchor_target_layer_tf.py中，并没有对anchors进行微调，直接对anchors求overlaps

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    # 求每个anchor和所有gt boxes的重叠范围
    # 返回所有anchors(N个)和所有gt boxes(K个)之间的IoU
    # 返回的是个N*K二维数组
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    # 取出每一行的最大值对应的列索引
    # 为每个anchor找到与它IoU最大的gt box索引
    # 这句话最重要的作用是为每个anchor选取一个gt box
    argmax_overlaps = overlaps.argmax(axis=1)
    # 取出每行最大的IoU
    # 结果是个一维向量，N个元素，每个anchor  和 gt box 最大的IoU值
    # 含义就是overlaps每行中的最大值
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    
    # 为每个gt box找到与它IoU最大的anchor索引
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    # 结果是个一维向量，K个元素，每个gt box  和 anchor 的IoU最大值
    # 含义就是overlaps每列中的最大值
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    # 找出overlaps中,值等于gt_max_oerlaps的索引
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them

        # IOU >= thresh: positive example
        # __C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
        # IOU < thresh: negative example
        # __C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    # 对于每个gt box来说，与他IoU最大的anchor的标签为fg
    # 下面这句是为gt box找与之最match的 anchor；  在proposal_target_layer中是:为proposal 找与之最match的gt box. 角度不同!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    # 对于每个anchor来说，如果它和某个gt box的IoU不低于0.7，那么该anchor的标签为fg
    # 下面这句是针对每个anchor而言,如果anchor和gt 的IoU >= 阈值则标记为fg的anchor
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1 # RPN_POSITIVE_OVERLAP 0.7

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0


    # 如果有超过128个正标签则对正标签进行采样
    # subsample positive labels if we have too many
    # RPN_BATCHSIZE=256 大于 rois BATCH_SIZE =128
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE) # 0.5*256 =128
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        # 从索引中多余数量的fg
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        # label从1变为-1，也就是忽视掉了这个fg标签
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    # 能接受的负标签的数量
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    # 当前负标签的数量
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # argmax_overlaps:overlaps每一行的最大值对应的列索引(也就是gt box的索引)
    # bbox_targets是anchor相对于对应的gt_rois的偏移量:tx,ty,tw,th
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # smooth L1的参数,对应论文中:4 parameterized coordinates of the predicted bounding box
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # Give the positive RPN examples weight of p * 1 / {num positives}
    # and give negatives a weight of (1 - p)
    # Set to -1.0 to use uniform example weighting
    # __C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

    # 如果RPN_POSITIVE_WEIGHT<0,就为bbox_outside_weights的正负例分配相同的初始化权重
    # 如果RPN_POSITIVE_WEIGHT>=0,分别为bbox_outside_weights的正负例分配初始化权重,权重是各自数量的倒数
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # 正例和负例标签的初始权重相同
        # uniform weighting of examples (given non-uniform sampling)
        # bg和fg的数量和
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    #pdb.set_trace()
    # A是最初anchor的数量,9
    # 下面这个labels有total_anchors个, 重新组织一下,回到最初的模样,feature map每个点有A个label
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    # ....这, 弄成caffe的那个blob
    labels = labels.reshape((1, 1, A * height, width))
    # (1, 1, A * height, width)
    rpn_labels = labels 

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

    # （1, A*4, height, width）
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    # (1, A*4, height, width)
    rpn_bbox_inside_weights = bbox_inside_weights


    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    # (1, A*4, height, width)
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights

'''
labels = _unmap(labels, total_anchors, inds_inside, fill=-1)； 
bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

这里主要是对total_anchors补充一些值，因为之前的操作都只是针对符合一定要求的anchor进行的
# keep only inside anchors
anchors = all_anchors[inds_inside, :]
'''
# 因为之前的操作只是针对符合一定要求的anchors进行的，还有很多anchors没有处理，所以这个函数主要是对total_anchors中没有处理过的anchors加上没有的值，比如labels，bbox_targets，bbox_inside_weights，bbox_outside_weights
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    # 如果data是一维的，进来的是labels
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    # data大于一维，进来的都是二维4列的
    else:
        # 每个位置元素都加4，下一步就都赋值为0了，为什么还要加4.。。？
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5
    
    # 输入rois和gt_rois        都是[x1,y1,x2,y2]
    # 输出各个rois相对于对应的gt_rois的偏移量
    # 4列分别代表：x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
