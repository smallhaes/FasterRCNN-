# coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# 计算用于训练Fast R-CNN network的minibatch blobs
"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


# 输入:roidb,物体类别数
# 输出: 图片的blob格式(字典). blob中包含的key有:gt_boxes,im_info,rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights
def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    # rois中有1/4的是fg rois
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    # 输入: roidb,scale_inds
    # 输出: 图片的blob形式;放缩倍数的列表
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        # 将原图中的boxes放缩到适合网络输入的尺寸
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    else: # not using RPN
        # 不使用RPN就得自己建立rois和对应的labels
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            # 将图片rois投影到训练用的图片尺寸
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        # 对bbox进行回归
        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

# 输入: roidb,每张图片fg roi的数量,每张图片roi的数量,物体类别
# 输出: labels, overlaps, rois, bbox_targets, bbox_inside_weights
# 输出: 最终留下的bg roi和fg roi:对应的标签,IoU,bbox目标值,bbox数组权重
def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    # 生成一个随机的rois例子,  这些rois由fg和bg的例子构成
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # roidb中的rois,不都选用,而是从中挑出bg rois和 fg rois
    # label = class RoI has max overlap with
    # 和roi最match的gt box对应的类别,相当于labels
    labels = roidb['max_classes']
    # roi和最match的gt box对应的最大IoU值
    overlaps = roidb['max_overlaps']
    # 各个box的坐标
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    # 选出rois中能作为fg roi的索引
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    # Element-wise minimum of array elements
    fg_rois_per_this_image = int(np.minimum(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    # 不放回地随机采样出fg_rois_per_this_image个fg_inds
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # overlaps在[BG_THRESH_LO, BG_THRESH_HI)这个范围的作为 bg roi
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    # 采样出bg_inds
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    # 留下的索引,由fg索引和bg索引构成,前面是fg 后面是 bg,这个顺序也要留意
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    # 提取出最终留下的索引对应的标签
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    # bg的标签为0
    labels[fg_rois_per_this_image:] = 0
    # 最终留下的索引对应的IoU (roi和其最match的gt roi的IoU,  roi包括bg roi 和 fg roi)
    overlaps = overlaps[keep_inds]
    # 最终索引对应的box坐标
    rois = rois[keep_inds]
    # roidb['bbox_targets']存储的是x,y,w,h这四种数据吧 ???   rois存的是x1,y1,x2,y2这四种
    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

# 输入: roidb,scale_inds
# 输出: 图片的blob形式;放缩倍数的列表
def _get_image_blob(roidb, scale_inds):
    # 将roidb中的图片按照特定的scale转换成blob格式
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            # 水平翻转,将width倒序排列即可
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]#SCALES 600
        # 对图片去均值并进行放缩,返回放缩后的图片以及放缩倍数
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE) # MAX_SIZE 1000
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    # 将图片rois投影到训练用的图片尺寸
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

# 输入: roidb中存放的rois对应的bbox的gt值( x,t,w,h); 类别数
# 输出: N行4K列数组:bbox_target_data(x,y,w,h);数组权重 bbox_inside_weights
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.  
    Bounding-box regression targets以紧凑的形式存放在roidb中
    本来有4K个元素,现在只用4个元素就OK了,因为剩下的元素都是0
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = np.array(bbox_target_data[:, 0], dtype=np.uint16, copy=True)
    # 最终选用的rois的数量这么多行, 4 * num_classes列
    # 每个class对应四4列, 对于每一行来说,只有对应类别所对应的四列有值:x,t,w,h,其余类别对应的四列都是0
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    # bbox_targets对应的内部权重?
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    # clss=0是bg类, clss>0是各个fg的类别
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        # BBOX_INSIDE_WEIGHTS都是1
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

# 可视化用的
def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
