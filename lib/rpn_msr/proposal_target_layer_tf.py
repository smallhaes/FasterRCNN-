# coding:utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import pdb

DEBUG = False

# 该函数具有三个功能
# 1.将proposals分配给ground-truth targets；
# 2.产生proposal classification labels(类别信息在gt boxes的第axis=4列) 
# 3.产生bounding-box regression targets (获得proposal相对与gt的tx,ty,tw,th)
# 输出: rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights
# 输出：采样得到的rois：rois_per_image个(0, x1, y1, x2, y2)，也就是128个(0, x1, y1, x2, y2)；rois对应的标签：128行一列，bbox_targets：N*(tx,ty,tw,th)，N=128；tx,ty,tw,th对应的权重参数inside；tx,ty,tw,th对应的权重参数outside
# rpn_rois:训练时2000个[0,x1,y1,x2,y2]，测试时300个[0,x1,y1,x2,y2]，其实这个函数只在训练时用到，所以传入了2000个
def proposal_target_layer(rpn_rois, gt_boxes,_num_classes):
    # _num_classes = 21
    # gt_boxes是个二维数组，一共5列，前4列是x1,y1,x2,y2
    """
    将proposals分配给gt targets， 生成proposal 分类标签和 bbox 回归targets
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    # 所有的proposals，训练时2000个
    all_rois = rpn_rois
    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

     
    # Include ground-truth boxes in the set of candidate rois
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    # gt_boxes可能有5列，前四列是x1,y1,x2,y2
    # 把gt boxes也纳入候选rois中， 这样做的目的是？  难道不应该只用输入的rpn_rois吗？？ 然后对这些rpn_rois进行回归。现在把gt也当成候选proposals了，这些proposal相对gt的偏差是0,也没法训练啊？？？？
    all_rois = np.vstack(
        (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    )

    # Sanity check: single batch only
    # 检查第0列都是0，不是0就报错
    # 但是这里是proposal啊？ 一次只处理一个proposal？
    assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

    num_images = 1 # batch_size = 1
    # 每张图片对应的rois
    # TRAIN.BATCH_SIZE ： Minibatch size (number of regions of interest [ROIs]) 这是rois的batch size！！
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images # 128/1
    # cfg.TRAIN.FG_FRACTION： 0.25
    # fg_rois_per_image： 128*0.25 = 32
    # FG_FRACTION：Fraction of minibatch that is labeled foreground (i.e. class > 0) rois的batch中fg所占的比例
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    # 输出： 最终proposal对应的labels(1维向量), 最终的proposals(rois):128个(0, x1, y1, x2, y2), bbox_targets：N*(tx,ty,tw,th), bbox_inside_weights:N*84
    labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)

    if DEBUG:
        print 'num fg: {}'.format((labels > 0).sum())
        print 'num bg: {}'.format((labels == 0).sum())
        _count += 1
        _fg_num += (labels > 0).sum()
        _bg_num += (labels == 0).sum()
        print 'num fg avg: {}'.format(_fg_num / _count)
        print 'num bg avg: {}'.format(_bg_num / _count)
        print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))
    # rois：128个(0, x1, y1, x2, y2)
    rois = rois.reshape(-1,5)
    # 128行1列的二维数组
    labels = labels.reshape(-1,1)
    # 不用reshape啊  本来就是4*21=84列  可能为了保险
    bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
    # 可能用两种weight在训练时效果更好
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)

    # 此时bbox_outside_weights，浮点数0.0与1.0
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    # 返回：采样得到的rois，rois对应的标签，rois相对其gt boxes的偏移量，偏移量对应的参数inside；偏移量对应的参数outside
    return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

# 功能：bbox_target_data:N*(cls,tx,ty,tw,th)调整成能够喂入网络的二维数组:N*(21*4)（one-hot思想）！
# 输入：bbox_target_data:N*(cls,tx,ty,tw,th)，num_classes=21
# 输出 :bbox_targets:N,21*4, bbox_inside_weights:N,21*4
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)，这是一种紧凑的排列方式，紧凑表现在只说明了对应class的信息，其余class的信息虽然没用，但是这种格式没法直接输入网络，所以要补充其余class的信息(全用0)

    将bbox_target_data这个二维数组扩展一下，仍是二维数组，但是不再是5列了，而是21*4=84列，每4列对应一个类别,其中只有proposal对应的gt 类别的4列不为0,其余20组4列全是0。 这其实体现了one-hot思想(one-hot能方便喂给网络！)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets；K=2
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    # 提取出类别，0是bg，1-20是不同物体的label！！！
    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    # num_classes = 21
    # 扩展成4*21=84列
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    # bbox_targets中tx,ty,tw,th对应的权重参数
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    # 找出fg的索引
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        # Deprecated (inside weights)
        #__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        # 为每个proposal的tx,ty,tw,th 分配权重参数
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

# 输入rois，gt_rois，labels
# 输出一个二维数组，
# 第0维是rois的数量；第1维的五列分别是：label,x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    #确保rois与gt_rois的数量一致
    assert ex_rois.shape[0] == gt_rois.shape[0]
    # 确保有x1,y1,x2,y2
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    # 输出各个rois相对于对应的gt_rois的偏移量
    # 4列分别代表：x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
    targets = bbox_transform(ex_rois, gt_rois)
    # 默认不normalization
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    # labels是个一维向量
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
# 采样rois
# 返回：采样出的rois的标签；采样出的rois；fg的rois与其对应的gt boxes的偏移量；bbox_targets矩阵对应的参数
# 返回：labels, rois, bbox_targets, bbox_inside_weights
# 输入中：all_rois是proposals和gt_boxes组成的，N+K个[0,x1,y1,x2,y2];gt_boxes 前4列是x1,y1,x2,y2,第5列是label
# 输出： 最终proposal对应的labels(1维向量), 最终的proposals:128个(0, x1, y1, x2, y2), bbox_targets：N*(tx,ty,tw,th), bbox_inside_weights:N*84
def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """
    从由fg和bg样本组成的rois中随机采样出一些样本
    Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    # 返回所有proposals(N个，2000个)和所有gt boxes(K个)之间的IoU
    # 返回的是个N*K二维数组
    # gt_boxes的前四列是x1,y1,x2,y2，第五列是label：bg，fg
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    # 对于每一个proposal，它与所有gt box的IoU中，返回最大IoU对应的gt box的索引
    gt_assignment = overlaps.argmax(axis=1)
    # 对于每一个proposal，它与所有gt box的IoU中，返回最大的IoU值
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # FG_THRESH = 0.5
    # Select foreground RoIs as those with >= FG_THRESH overlap
    # np.where返回的是proposals的索引，这些索引对应的IoU满足>fg_thresh的条件，所以也是fg索引了
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    # 不放回采样
    if fg_inds.size > 0:
        # 从fg_inds中不放回随机采样fg_rois_per_this_image个rois
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    # 0.1 < max_overlaps <= 0.5 的作为bg
    # np.where返回的是proposals的索引，这些索引对应的IoU满足bg不等式条件，所以也是bg索引了
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    # 保证 fg + bg 的数量小于等于 rois_per_image(每张图片的roi数量)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        # 从bg_inds中不放回采样bg_rois_per_this_image个rois
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    # 这里是np.append(),对于一维array，相当于list.extend()
    keep_inds = np.append(fg_inds, bg_inds) # 一共是rois_per_image个：128
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    # 将bg的labels 设为 0
    labels[fg_rois_per_this_image:] = 0
    # 128个
    rois = all_rois[keep_inds]
    # 上面两句是最终的rois和rois对应的标签：1是fg，0是bg
# =======================================================

    # gt_assignment存的是根据IoU初步选出gt_boxes索引，gt_assignment = overlaps.argmax(axis=1)
    # gt_assignment[keep_inds]是最终选出的proposals对应的gt_boxes索引
    # 计算最终的proposals和其gt boxes之间的偏移量；同时将每个proposal对应的标签放到第0列
    #返回2维数组，第0列是proposal的label，第1-4列是proposal与gt boxes之间的偏移量：x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    # Returns:
    # bbox_target (ndarray): N x 4K blob of regression targets；K=2
    # bbox_inside_weights (ndarray): N x 4K blob of loss weights；bbox_target对应的参数
    # 输出 :bbox_targets:N,21*4, bbox_inside_weights:N,21*4
    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
