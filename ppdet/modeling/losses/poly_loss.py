from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle

from ppdet.core.workspace import register, serializable

__all__ = ['PolyLoss']


@register
@serializable
class PolyLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, ppoly, gpoly):
        px1, py1, px2, py2, px3, py3, px4, py4 = ppoly
        gx1, gy1, gx2, gy2, gx3, gy3, gx4, gy4 = gpoly

        x1 = paddle.abs(px1 - gx1)
        y1 = paddle.abs(py1 - gy1)
        x2 = paddle.abs(px2 - gx2)
        y2 = paddle.abs(py2 - gy2)
        x3 = paddle.abs(px3 - gx3)
        y3 = paddle.abs(py3 - gy3)
        x4 = paddle.abs(px4 - gx4)
        y4 = paddle.abs(py4 - gy4)

        iou = (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4) / 1000.

        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou
