# coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

# 输入rois/anchors和gt_rois        都是[x1,y1,x2,y2]
# 输出各个rois/anchors相对于对应的gt_rois的偏移量
# 4列分别代表：x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
def bbox_transform(ex_rois, gt_rois):
    # x2 - x1 + 1
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    # y2 - y1 + 1
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    # x1 + 0.5w
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    # y1 + 0.5h
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    # 对应论文的四个公式，越接近0说明和gt越靠近
    # 尺度无关的偏移量，尺度无关的伸缩量
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    # 4列分别代表：x的偏移量,y的偏移量,w的伸缩量,h的伸缩量
    return targets

# bbox_transform_inv(anchors, bbox_deltas)
# anchors:(K*A, 4)
# bbox_deltas:(1 * H * W * A, 4) where rows are ordered by (h, w, a)
# 输入：一张图片中所有的anchors及模型预测出的所有anchors的dx,dy,dw,dh(或者说featuremap所有点对应的所有anchors的dx,dy,dw,dh)
# 输出：anchors修正后的坐标[x1,y1,x2,y2]（也就是proposals了！），4列分别代表：x,y,w,h
def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    # x2 - x1 + 1
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    # y2 - y1 + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    # x1 + 0.5w
    ctr_x = boxes[:, 0] + 0.5 * widths
    # y1 + 0.5h
    ctr_y = boxes[:, 1] + 0.5 * heights

    # 取出deltas的第一列
    dx = deltas[:, 0::4]
    # 取出deltas的第二列
    dy = deltas[:, 1::4]
    # 取出deltas的第三列
    dw = deltas[:, 2::4]
    # 取出deltas的第四列
    dh = deltas[:, 3::4]

    # 中心点的偏移
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # 尺度的缩放
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # 转换后的x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # 转换后的y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # 转换后的x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # 转换后的y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

# clip_boxes(proposals, im_info[:2])
# 输入proposals，图片的height和width
# 返回clip后的proposals[x1,y1,x2,y2]
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # im_shape[1]是图片的width
    # im_shape[0]是图片的height
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
