# coding:utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
import pdb


DEBUG = False
"""
对anchors施加变换得到proposals
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
#
# 返回： 训练时返回2000个[0,x1,y1,x2,y2]，测试时返回300个[0,x1,y1,x2,y2]
def proposal_layer(rpn_cls_prob_reshape,rpn_bbox_pred,im_info,cfg_key,_feat_stride = [16,],anchor_scales = [8, 16, 32]):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold 去掉宽高小于阈值的predicted boxes
    # sort all (proposal, score) pairs by score from highest to lowest 将(proposal, score)对进行降序排序
    # take top pre_nms_topN proposals before NMS 选出前pre_nms_topN个proposals
    # apply NMS with threshold 0.7 to remaining proposals 对当前的proposals进行NMS
    # take after_nms_topN proposals after NMS 选出前after_nms_topN个proposals作为最后的proposals
    # return the top proposals (-> RoIs top, scores top) 
    #layer_params = yaml.load(self.param_str_)


    # _anchors: 9*4  9代表9个anchor, 4代表[x1,y1,x2,y2]
    # def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6))
    # 返回了9种size的anchors:3*3;    论文中是9种:3*3
    # 最重要的是要对着9种anchor进行各种平移,使得这些anchors能遍布原图的各个地方!!
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    # 9个anchor;
    _num_anchors = _anchors.shape[0] 


    # rpn_cls_prob_reshape:(1, 18, H, W)
    rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2]) # [0],d,[1]/d*[3],[2] 
    rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2])      # (1, 4 * A, H, W)
    #rpn_cls_prob_reshape = np.transpose(np.reshape(rpn_cls_prob_reshape,[1,rpn_cls_prob_reshape.shape[0],rpn_cls_prob_reshape.shape[1],rpn_cls_prob_reshape.shape[2]]),[0,3,2,1])
    #rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,2,1])
    im_info = im_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    #cfg_key = 'TEST'
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N # train时12000,test时6000
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # train时2000,test时300
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH # 0.7
    min_size      = cfg[cfg_key].RPN_MIN_SIZE  # 16

    # 第一类(前 _num_anchors个)是bg背景类(非物体类)
    # 第二类是fg前景类(物体类)
    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # rpn_cls_prob_reshape的第1维有18个数,前9个表示bg的anchors,后9个表示fg的anchors
    #  _num_anchors:       表示取出后9个anchors
    # scores:(1, A, H, W)
    # fg的scores就是经过softmax后的概率值, 之后会根据fg的scores/概率大小降序排序,取出前面的fg
    # scores: (1, 9, H, W)
    # 时刻注意:这里的score对应的都是fg的!!!
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]

    # 9个anchor,每个anchor对应4个值:x,y,w,h
    bbox_deltas = rpn_bbox_pred # (1, 4 * A, H, W)
    #im_info = bottom[2].data[0, :]

    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])

    # 1.从bbox deltas和shifted anchors中生成proposals
    # 1. Generate proposals from bbox deltas and shifted anchors
    # height:高度
    # width:宽度
    # scores: (1, 9, H, W)
    height, width = scores.shape[-2:]

    if DEBUG:
        print 'score map size: {}'.format(scores.shape)

    # Enumerate all shifts
    # 这里应该是为找出对应原图的anchors，先找出anchors的平移量？
    # _feat_stride = [16,]
    # feature map的点恢复到原图大小的规模
    shift_x = np.arange(0, width) * _feat_stride 
    shift_y = np.arange(0, height) * _feat_stride 
    # shift_x是axis-wise级别的重复, shift_y是element-wise级别的重复
    # 返回的shift_x和shift_y可以构成很多个平移对儿:(x的平移量,y的平移量)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)#形式类似这样: shift_x: [[0, 16, 32],[0, 16, 32],[0, 16, 32]]    shift_y: [0,0,0,16,16,16,32,32,32],实际上有更多元素
    # ravel函数是将矩阵变为一个一维的数组
    # 将这四个array竖直组合后再转置后便会生成很多组平移对儿(x1的平移量,y1的平移量,x2的平移量,y2的平移量)
    # 这种构造方法很巧妙! 接下来将这种变换施加到各个点对应的9个feature map上就能得到所有的anchors了(当然了,这些anchors太多了,所以最后还要进行筛选)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors # 9
    K = shifts.shape[0] # 所有平移对儿的数量,这个数量由height, width = scores.shape[-2:]决定
    # 枚举出所有anchors
    # 这里便是对9个anchors施加各种平移后得到的所有anchors
    # 这个np加法也很有讲究
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order

    # rpn_bbox_pred所有点对应的所有anchors 
    # 最重要的是:bbox_deltas模型预测出的!!!!!!  而且预测的不是x,y,w,h,而是dx,dy,dw,dh !!!!!!!   要和之前使用各种平移对儿得到的anchors区分开!!!!!!
    # 测试时将dx,dy,dw,dh加在哪些anchors上???(在这里叫作bbox_deltas)
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    # 时刻注意:这里的score对应的都是fg的!!!
    # scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :] # scores: (1, 9, H, W)
    # 将rpn_cls_prob_reshape所有点对应的fg得分排成一列
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))


    # 通过bbox transformations 将anchors 转换成proposals
    # Convert anchors into proposals via bbox transformations
    # 传入anchors和模型预测出的bbox_deltas，返回proposals:[x1,y1,x2,y2]
    # 在这个函数中,将会看到anchors的作用. 生成anchors时,是生成9anchors,再进行各种各样的平移得到更多的anchors; 生成bbox_deltas时,是为feature map的每个点生成9个anchors的dx,dy,dw,dh!!!!!!
    # 为所有的anchors加上bbox_deltas。 这个函数贼重要，修正了anchors！！
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    # im_info[0]是height
    # im_info[1]是weight
    # 裁剪一下proposals, 是proposals的x1,y1,x2,y2都在图片范围内
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    # 注意要将min_size转换到原图大小的尺度上
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    # im_info[2]是feature map 到原图片的放大倍数：16
    keep = _filter_boxes(proposals, min_size * im_info[2])
    # 留下满足ws和hs均大于min_size的proposals及对应的scores
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    # order是降序排序的元素的索引
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    # 返回的也是索引
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        # train时2000,test时300
        keep = keep[:post_nms_topN] 
    proposals = proposals[keep, :]
    scores = scores[keep]
    
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    # 第0维都是0
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob
    #top[0].reshape(*(blob.shape))
    #top[0].data[...] = blob

    # [Optional] output scores blob
    #if len(top) > 1:
    #    top[1].reshape(*(scores.shape))
    #    top[1].data[...] = scores

# _filter_boxes(proposals, min_size * im_info[2])
# 输入：proposals和边的最小size
# 返回boxes中满足ws和hs均大于min_size的索引
def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    # x2 - x1 + 1
    ws = boxes[:, 2] - boxes[:, 0] + 1
    # y2 - y1 + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    # 返回满足ws和hs均大于min_size的索引
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
