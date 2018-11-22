# coding:utf-8
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# 训练Fast R-CNN网络
"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
# 下面这两个roidb有什么不同？
# 虽说是gt_data_layer，但也有不是gt 的anchors产生
# 为imdb.roidb的每个元素(每个元素都是字典)添加新的信息
# 添加键'info_boxes'，对应的value是个二维数组，共18列，由(cx, cy, scale_ind, box, scale_ind_map, box_map, gt_label, gt_sublabel, target)
import gt_data_layer.roidb as gdl_roidb # 生成anchors， 训练时采用的是gdl，说明用其他函数生成的anchors？
# 丰富imdb中的roidb
# 这个比较迷，看具体用法吧
# 为imdb.roidb的每个元素(每个元素都是字典)添加新的信息
# 添加键:image,width,height,max_classes,max_overlaps
import roi_data_layer.roidb as rdl_roidb # 并没有生成anchors
# 获得minibatch数据,用于训练Fast R-CNN network
from roi_data_layer.layer import RoIDataLayer

from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
from tensorflow.core.protobuf import saver_pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver. 一个关于Caffe求解程序的简单包装程序
    This wrapper gives us control over the snapshotting process(快照过程), which we
    use to unnormalize(非标准化) the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = saver

    # 功能：将unnormalizing的回归权重存起来
    def snapshot(self, sess, iter):
        # 在未规范化学习的限制框回归权值后，对网络进行快照。
        # 在对学习的边界框回归权重进行非标准化后拍摄网络快照
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            # 获取weights的具体值
            orig_0 = weights.eval()
            # 获取biases的具体值
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            # 这种相乘是element-wise的，具体来说就是对应元素相乘，并不是矩阵相乘
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})

            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # infix：中缀
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                # 将网络恢复到原来的状态
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        # 作为x;      注意理解参数化过程,原论文中的parameterized coordinates
        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    # 这里的参数是不是写成self.sess,self.max_iters更好些
    def train_model(self, sess, max_iters):
    #=======================================================================================
    # 静态图构建：数据(placeholder)，网络，优化器
    #=======================================================================================
        """Network training loop."""

        # 将处理好的数据组织一下，准备喂给网络，或者说用来构造feed_dict
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # RPN
        # classification loss
        # 返回名为'rpn_cls_score_reshape'的层的输出，并进行reshape
        # 现在的shape:[A *height *width,2]                 [0],[1]/d*[3],[2],d  
          
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        # 现在的shape:[A * height *width],一维的
        # 256,1
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        
        # 取出bg和fg的scores
        # # 256,2  
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        # 取出label是1或0的,忽略label是-1的
        # 忽略-1后剩下256个,说明1,0两种标签一共256个, 但是如何确定标签对应feature map哪个点? 毕竟这些标签是通过anchor得到的,而anchor的中心位置是确定的.
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))




        # bounding box regression L1 loss
        # 现在的shape是(1,h,w,36), 这里面的像素点应该和rpn-data的点一一对应
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        # 这个是gt
        # 现在的shape是(1, height, width,A*4）   A=9
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])
        # def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
        rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))
 
        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]

        smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
        loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

        # final loss
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                        cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        momentum = cfg.TRAIN.MOMENTUM

        # 这是启动训练的钥匙！
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)

        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()


    #=========================================================================================
    # 正式训练过程:数据(feed_dict),sess.run
    #=========================================================================================
        for iter in range(max_iters):
            # get one batch
            # 先获得一个batch数据,再用这个数据构造feed_dict
            blobs = data_layer.forward()

            # Make one SGD update
            feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
                           self.net.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            # 训练计时    
            timer.tic()
            # 返回这些loss方便观察（并没有用tensorboard，用tensorboard的话更方便观察损失变化趋势）
            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op],
                                                                                                feed_dict=feed_dict,
                                                                                                options=run_options,
                                                                                                run_metadata=run_metadata)

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())

                # 这种计时方法很好！
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


# 输入：imdb
# 输出：为imdb.roi的每个元素添加信息，返回roidb用于训练
# roidb是个列表，每个元素都代表一张图片，元素的内容是包含图片信息的字典，这些信息包括：boxes，gt_overlaps，gt_classes，flipped
def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    # 训练时是否使用RPN
    if cfg.TRAIN.HAS_RPN:
        # 是否多尺度
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    # 返回imdb中最关键的属性roidb，此时roidb的每个元素包含更多的信息，视gdl或rdl而定
    return imdb.roidb

# 返回一个数据层

def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

# 输入：roidb
# 输出：返回过滤后的roidb(去除了roidb中无用的元素，具体来说就是overlaps不满足阈值要求)
def filter_roidb(roidb):
    # 去除拥有无用的roi的imdb.roidb 条目
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


# 输入的roidb是扩充信息后的imdb.roidb
def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    # Saves and restores variables
    saver = tf.train.Saver(max_to_keep=100,write_version=saver_pb2.SaverDef.V1)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # 训练时，network = networks.VGGnet_train()
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
