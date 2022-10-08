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

import os
import sys
import json
import paddle
import numpy as np

from .map_utils import prune_zero_padding, DetectionMAP
from .coco_utils import get_infer_results, cocoapi_eval
from .widerface_utils import face_eval_run
from ppdet.data.source.category import get_categories

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'Metric',
    'COCOMetric',
    'VOCMetric',
    'PolyVOCMetric',
    'WiderFaceMetric',
    'get_infer_results',
    'RBoxMetric',
    'SNIPERCOCOMetric'
]

COCO_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87,
    .89, .89
]) / 10.0
CROWD_SIGMAS = np.array(
    [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79,
     .79]) / 10.0


class Metric(paddle.metric.Metric):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    # paddle.metric.Metric defined :metch:`update`, :meth:`accumulate`
    # :metch:`reset`, in ppdet, we also need following 2 methods:

    # abstract method for logging metric results
    def log(self):
        pass

    # abstract method for getting metric results
    def get_results(self):
        pass


class COCOMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        self.anno_file = anno_file
        self.clsid2catid = kwargs.get('clsid2catid', None)
        if self.clsid2catid is None:
            self.clsid2catid, _ = get_categories('COCO', anno_file)
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.iou_type = kwargs.get('IouType', 'bbox')
        self.reset()

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.results['bbox'] += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        self.results['mask'] += infer_results[
            'mask'] if 'mask' in infer_results else []
        self.results['segm'] += infer_results[
            'segm'] if 'segm' in infer_results else []
        self.results['keypoint'] += infer_results[
            'keypoint'] if 'keypoint' in infer_results else []

    def accumulate(self):
        if len(self.results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                logger.info('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                bbox_stats = cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

        if len(self.results['mask']) > 0:
            output = "mask.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['mask'], f)
                logger.info('The mask result is saved to mask.json.')

            if self.save_prediction_only:
                logger.info('The mask result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['segm']) > 0:
            output = "segm.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['segm'], f)
                logger.info('The segm result is saved to segm.json.')

            if self.save_prediction_only:
                logger.info('The segm result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.results['keypoint']) > 0:
            output = "keypoint.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.results['keypoint'], f)
                logger.info('The keypoint result is saved to keypoint.json.')

            if self.save_prediction_only:
                logger.info('The keypoint result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                style = 'keypoints'
                use_area = True
                sigmas = COCO_SIGMAS
                if self.iou_type == 'keypoints_crowd':
                    style = 'keypoints_crowd'
                    use_area = False
                    sigmas = CROWD_SIGMAS
                keypoint_stats = cocoapi_eval(
                    output,
                    style,
                    anno_file=self.anno_file,
                    classwise=self.classwise,
                    sigmas=sigmas,
                    use_area=use_area)
                self.eval_results['keypoint'] = keypoint_stats
                sys.stdout.flush()

    def log(self):
        pass

    def get_results(self):
        return self.eval_results


class VOCMetric(Metric):
    def __init__(self,
                 label_list,
                 class_num=20,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False):
        assert os.path.isfile(label_list), \
                "label_list {} not a file".format(label_list)
        self.clsid2catid, self.catid2name = get_categories('VOC', label_list)

        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult,
            catid2name=self.catid2name,
            classwise=classwise)

        self.reset()

    def reset(self):
        self.detection_map.reset()

    def update(self, inputs, outputs):
        bbox_np = outputs['bbox'].numpy()
        bboxes = bbox_np[:, 2:]
        scores = bbox_np[:, 1]
        labels = bbox_np[:, 0]
        bbox_lengths = outputs['bbox_num'].numpy()

        if bboxes.shape == (1, 1) or bboxes is None:
            return
        gt_boxes = inputs['gt_bbox']
        gt_labels = inputs['gt_class']
        difficults = inputs['difficult'] if not self.evaluate_difficult \
                            else None

        scale_factor = inputs['scale_factor'].numpy(
        ) if 'scale_factor' in inputs else np.ones(
            (gt_boxes.shape[0], 2)).astype('float32')

        bbox_idx = 0
        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i].numpy()
            h, w = scale_factor[i]
            gt_box = gt_box / np.array([w, h, w, h])
            gt_label = gt_labels[i].numpy()
            difficult = None if difficults is None \
                            else difficults[i].numpy()
            bbox_num = bbox_lengths[i]
            bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
            score = scores[bbox_idx:bbox_idx + bbox_num]
            label = labels[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                             difficult)
            self.detection_map.update(bbox, score, label, gt_box, gt_label,
                                      difficult)
            bbox_idx += bbox_num

    def accumulate(self):
        logger.info("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                       self.map_type, map_stat))

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}


class PolyVOCMetric(Metric):
    def __init__(self,
                 label_list,
                 class_num=20,
                 overlap_thresh=0.5,
                 map_type='11point',
                 is_bbox_normalized=False,
                 evaluate_difficult=False,
                 classwise=False):
        assert os.path.isfile(label_list), \
                "label_list {} not a file".format(label_list)
        self.clsid2catid, self.catid2name = get_categories('PolyVOC', label_list)

        self.overlap_thresh = overlap_thresh
        self.map_type = map_type
        self.evaluate_difficult = evaluate_difficult
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=overlap_thresh,
            map_type=map_type,
            is_bbox_normalized=is_bbox_normalized,
            evaluate_difficult=evaluate_difficult,
            catid2name=self.catid2name,
            classwise=classwise)

        self.reset()

    def reset(self):
        self.detection_map.reset()

    def update(self, inputs, outputs):
        bbox_np = outputs['poly'].numpy()
        # bboxes = bbox_np[:, 2:6]
        polyes = bbox_np[:, 2:]
        bboxes = np.zeros((polyes.shape[0], 4))
        bboxes[..., 0] = (polyes[..., 0] + polyes[..., 2] + polyes[..., 4] + polyes[..., 6]) / 4
        bboxes[..., 1] = (polyes[..., 1] + polyes[..., 3] + polyes[..., 5] + polyes[..., 7]) / 4
        for i in range(polyes.shape[0]):
            xmin = min(polyes[i, 0], polyes[i, 2], polyes[i, 4], polyes[i, 6])
            ymin = min(polyes[i, 1], polyes[i, 3], polyes[i, 5], polyes[i, 7])
            xmax = max(polyes[i, 0], polyes[i, 2], polyes[i, 4], polyes[i, 6])
            ymax = max(polyes[i, 1], polyes[i, 3], polyes[i, 5], polyes[i, 7])
            bboxes[i, 2] = xmax - xmin
            bboxes[i, 3] = ymax - ymin
        scores = bbox_np[:, 1]
        labels = bbox_np[:, 0]
        bbox_lengths = outputs['bbox_num'].numpy()

        if bboxes.shape == (1, 1) or bboxes is None:
            return
        gt_polyes = inputs['gt_poly']
        gt_labels = inputs['gt_class']
        difficults = inputs['difficult'] if not self.evaluate_difficult \
                            else None

        scale_factor = inputs['scale_factor'].numpy(
        ) if 'scale_factor' in inputs else np.ones(
            (gt_polyes.shape[0], 2)).astype('float32')

        bbox_idx = 0
        for i in range(len(gt_polyes)):
            gt_poly = gt_polyes[i].numpy()
            h, w = scale_factor[i]
            gt_poly = gt_poly / np.array([w, h, w, h, w, h, w, h])
            gt_box = np.zeros((gt_poly.shape[0], 4))
            gt_box[..., 0] = (gt_poly[..., 0] + gt_poly[..., 2] + gt_poly[..., 4] + gt_poly[..., 6]) / 4
            gt_box[..., 1] = (gt_poly[..., 1] + gt_poly[..., 3] + gt_poly[..., 5] + gt_poly[..., 7]) / 4
            for j in range(gt_poly.shape[0]):
                xmin = min(gt_poly[j, 0], gt_poly[j, 2], gt_poly[j, 4], gt_poly[j, 6])
                ymin = min(gt_poly[j, 1], gt_poly[j, 3], gt_poly[j, 5], gt_poly[j, 7])
                xmax = max(gt_poly[j, 0], gt_poly[j, 2], gt_poly[j, 4], gt_poly[j, 6])
                ymax = max(gt_poly[j, 1], gt_poly[j, 3], gt_poly[j, 5], gt_poly[j, 7])
                gt_box[j, 2] = xmax - xmin
                gt_box[j, 3] = ymax - ymin
            gt_label = gt_labels[i].numpy()
            difficult = None if difficults is None \
                            else difficults[i].numpy()
            bbox_num = bbox_lengths[i]
            bbox = bboxes[bbox_idx:bbox_idx + bbox_num]
            score = scores[bbox_idx:bbox_idx + bbox_num]
            label = labels[bbox_idx:bbox_idx + bbox_num]
            gt_box, gt_label, difficult = prune_zero_padding(gt_box, gt_label,
                                                             difficult)
            
            self.detection_map.update(bbox, score, label, gt_box, gt_label,
                                      difficult)
            bbox_idx += bbox_num

    def accumulate(self):
        logger.info("Accumulating evaluatation results...")
        self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                       self.map_type, map_stat))

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}


class WiderFaceMetric(Metric):
    def __init__(self, image_dir, anno_file, multi_scale=True):
        self.image_dir = image_dir
        self.anno_file = anno_file
        self.multi_scale = multi_scale
        self.clsid2catid, self.catid2name = get_categories('widerface')

    def update(self, model):

        face_eval_run(
            model,
            self.image_dir,
            self.anno_file,
            pred_dir='output/pred',
            eval_mode='widerface',
            multi_scale=self.multi_scale)


class RBoxMetric(Metric):
    def __init__(self, anno_file, **kwargs):
        assert os.path.isfile(anno_file), \
                "anno_file {} not a file".format(anno_file)
        assert os.path.exists(anno_file), "anno_file {} not exists".format(
            anno_file)
        self.anno_file = anno_file
        self.gt_anno = json.load(open(self.anno_file))
        cats = self.gt_anno['categories']
        self.clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
        self.catid2clsid = {cat['id']: i for i, cat in enumerate(cats)}
        self.catid2name = {cat['id']: cat['name'] for cat in cats}
        self.classwise = kwargs.get('classwise', False)
        self.output_eval = kwargs.get('output_eval', None)
        # TODO: bias should be unified
        self.bias = kwargs.get('bias', 0)
        self.save_prediction_only = kwargs.get('save_prediction_only', False)
        self.iou_type = kwargs.get('IouType', 'bbox')
        self.overlap_thresh = kwargs.get('overlap_thresh', 0.5)
        self.map_type = kwargs.get('map_type', '11point')
        self.evaluate_difficult = kwargs.get('evaluate_difficult', False)
        class_num = len(self.catid2name)
        self.detection_map = DetectionMAP(
            class_num=class_num,
            overlap_thresh=self.overlap_thresh,
            map_type=self.map_type,
            is_bbox_normalized=False,
            evaluate_difficult=self.evaluate_difficult,
            catid2name=self.catid2name,
            classwise=self.classwise)

        self.reset()

    def reset(self):
        self.result_bbox = []
        self.detection_map.reset()

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        infer_results = get_infer_results(
            outs, self.clsid2catid, bias=self.bias)
        self.result_bbox += infer_results[
            'bbox'] if 'bbox' in infer_results else []
        bbox = [b['bbox'] for b in self.result_bbox]
        score = [b['score'] for b in self.result_bbox]
        label = [b['category_id'] for b in self.result_bbox]
        label = [self.catid2clsid[e] for e in label]
        gt_box = [
            e['bbox'] for e in self.gt_anno['annotations']
            if e['image_id'] == outs['im_id']
        ]
        gt_label = [
            e['category_id'] for e in self.gt_anno['annotations']
            if e['image_id'] == outs['im_id']
        ]
        gt_label = [self.catid2clsid[e] for e in gt_label]
        self.detection_map.update(bbox, score, label, gt_box, gt_label)

    def accumulate(self):
        if len(self.result_bbox) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.result_bbox, f)
                logger.info('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                logger.info('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                logger.info("Accumulating evaluatation results...")
                self.detection_map.accumulate()

    def log(self):
        map_stat = 100. * self.detection_map.get_map()
        logger.info("mAP({:.2f}, {}) = {:.2f}%".format(self.overlap_thresh,
                                                       self.map_type, map_stat))

    def get_results(self):
        return {'bbox': [self.detection_map.get_map()]}


class SNIPERCOCOMetric(COCOMetric):
    def __init__(self, anno_file, **kwargs):
        super(SNIPERCOCOMetric, self).__init__(anno_file, **kwargs)
        self.dataset = kwargs["dataset"]
        self.chip_results = []

    def reset(self):
        # only bbox and mask evaluation support currently
        self.results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}
        self.chip_results = []

    def update(self, inputs, outputs):
        outs = {}
        # outputs Tensor -> numpy.ndarray
        for k, v in outputs.items():
            outs[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

        im_id = inputs['im_id']
        outs['im_id'] = im_id.numpy() if isinstance(im_id,
                                                    paddle.Tensor) else im_id

        self.chip_results.append(outs)


    def accumulate(self):
        results = self.dataset.anno_cropper.aggregate_chips_detections(self.chip_results)
        for outs in results:
            infer_results = get_infer_results(outs, self.clsid2catid, bias=self.bias)
            self.results['bbox'] += infer_results['bbox'] if 'bbox' in infer_results else []

        super(SNIPERCOCOMetric, self).accumulate()
