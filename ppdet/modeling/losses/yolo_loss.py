# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..bbox_utils import decode_yolo, xywh2xyxy, iou_similarity, decode_poly_yolo, xywh2xyxy_poly

__all__ = ['YOLOv3Loss']


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox

def poly_transform(pbox, anchor, downsample):
    pbox = decode_poly_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


@register
class YOLOv3Loss(nn.Layer):

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance  
        """
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        # pbox
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gx1y1 = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gx2y2 = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = paddle.concat([gx1y1, gx2y2], axis=-1)

        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])

        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj
        loss = dict()

        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            loss_x = F.binary_cross_entropy(x, tx, reduction='none')
            loss_y = F.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = paddle.abs(x - tx)
            loss_y = paddle.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()

        loss_w = paddle.abs(w - tw)
        loss_h = paddle.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh

        if self.iou_loss is not None:
            # warn: do not modify x, y, w, h in place
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss['loss_iou'] = loss_iou

        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware

        box = [x, y, w, h]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_box = targets['gt_bbox']
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(inputs, gt_targets, anchors,
                                            self.downsample):
            yolo_loss = self.yolov3_loss(x, t, gt_box, anchor, downsample,
                                         self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss
        return yolo_losses


@register
class YOLOPolyLoss(nn.Layer):

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):
        """
        YOLOPolyLoss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance  
        """
        super(YOLOPolyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        # pbox
        pbox = poly_transform(pbox, anchor, downsample)
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gx1y1 = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gx2y2 = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = paddle.concat([gx1y1, gx2y2], axis=-1)

        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    def yolo_poly_loss(self, p, t, gt_poly, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        gt_box = gt_poly[:, :, 0:4]
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w1, h1 = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        w2, h2 = p[:, :, :, :, 4:5], p[:, :, :, :, 5:6]
        w3, h3 = p[:, :, :, :, 6:7], p[:, :, :, :, 7:8]
        w4, h4 = p[:, :, :, :, 8:9], p[:, :, :, :, 9:10]
        obj, pcls = p[:, :, :, :, 10:11], p[:, :, :, :, 11:]
        self.distill_pairs.append([x, y, w1, h1, w2, h2, w3, h3, w4, h4, obj, pcls])

        w1 = F.sigmoid(w1)
        h1 = F.sigmoid(h1)
        w2 = F.sigmoid(w2)
        h2 = F.sigmoid(h2)
        w3 = F.sigmoid(w3)
        h3 = F.sigmoid(h3)
        w4 = F.sigmoid(w4)
        h4 = F.sigmoid(h4)

        gw_max = paddle.maximum(paddle.maximum(w1, w2), paddle.maximum(w3, w4))
        gw_min = paddle.minimum(paddle.minimum(w1, w2), paddle.minimum(w3, w4))
        gh_max = paddle.maximum(paddle.maximum(h1, h2), paddle.maximum(h3, h4))
        gh_min = paddle.minimum(paddle.minimum(h1, h2), paddle.minimum(h3, h4))
        gw = paddle.subtract(gw_max, gw_min)
        gh = paddle.subtract(gh_max, gh_min)

        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw1, th1 = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tw2, th2 = t[:, :, :, :, 4:5], t[:, :, :, :, 5:6]
        tw3, th3 = t[:, :, :, :, 6:7], t[:, :, :, :, 7:8]
        tw4, th4 = t[:, :, :, :, 8:9], t[:, :, :, :, 9:10]
        tscale = t[:, :, :, :, 10:11]
        tobj, tcls = t[:, :, :, :, 11:12], t[:, :, :, :, 12:]

        tw_max = paddle.maximum(paddle.maximum(tw1, tw2), paddle.maximum(tw3, tw4))
        tw_min = paddle.minimum(paddle.minimum(tw1, tw2), paddle.minimum(tw3, tw4))
        th_max = paddle.maximum(paddle.maximum(th1, th2), paddle.maximum(th3, th4))
        th_min = paddle.minimum(paddle.minimum(th1, th2), paddle.minimum(th3, th4))
        tw = paddle.subtract(tw_max, tw_min)
        th = paddle.subtract(th_max, th_min)

        tscale_obj = tscale * tobj
        loss = dict()

        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            loss_x = F.binary_cross_entropy(x, tx, reduction='none')
            loss_y = F.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = paddle.abs(x - tx)
            loss_y = paddle.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()

        loss_w = paddle.abs(gw - tw)
        loss_h = paddle.abs(gh - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss_w1 = paddle.abs(w1 - tw1)
        loss_h1 = paddle.abs(h1 - th1)
        loss_w2 = paddle.abs(w2 - tw2)
        loss_h2 = paddle.abs(h2 - th2)
        loss_w3 = paddle.abs(w3 - tw3)
        loss_h3 = paddle.abs(h3 - th3)
        loss_w4 = paddle.abs(w4 - tw4)
        loss_h4 = paddle.abs(h4 - th4)
        loss_ponit = tscale_obj * (loss_w1 + loss_h1 + loss_w2 + loss_h2 + loss_w3 + loss_h3 + loss_w4 + loss_h4) / 4.0 * 1.2
        loss_ponit = loss_ponit.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        # loss['loss_wh'] = loss_wh
        loss['loss_ponit'] = loss_ponit

        if self.iou_loss is not None:
            # warn: do not modify x, y, w, h in place
            box, tbox = [x, y, gw, gh], [tx, ty, tw, th]
            pbox = poly_transform(box, anchor, downsample)
            gbox = poly_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss['loss_iou'] = loss_iou

        if self.iou_aware_loss is not None:
            box, tbox = [x, y, gw, gh], [tx, ty, tw, th]
            pbox = poly_transform(box, anchor, downsample)
            gbox = poly_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware

        box = [x, y, gw, gh]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_poly = targets['gt_poly']
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(inputs, gt_targets, anchors,
                                            self.downsample):
            yolo_loss = self.yolo_poly_loss(x, t, gt_poly, anchor, downsample,
                                         self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss
        return yolo_losses
