# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..self_supervision_scramble import SelfSup_Scrambler
import random
from ..roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
import tensorboardX
from ...data.transforms.transforms import DeNormalize
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList

from PIL.ImageDraw import Draw
from torchvision.transforms import ToPILImage

import cv2, copy

def meta_obtain_pseudo_labels(self, images, features, params=None):
    self.eval()

    test_proposals,_ = self.rpn(images, features, params=self.get_subdict(params, "rpn"))
    _, pseudo_targets, _ = self.roi_heads(features, test_proposals, params=self.get_subdict(params, "roi_heads"))

    if len(pseudo_targets[0]) == 0:
       print("No pseudo targets!")
    self.train()
    return pseudo_targets

def grcnn_forward(self, images, targets=None, auxiliary_task=False, params=None, meta = False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
            auxiliary_task (Bool): if the auxiliary task is enabled during training

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        if self.training and targets is None and not auxiliary_task:
            raise ValueError("In training mode, targets should be passed")

        if not self.training and auxiliary_task:
            raise ValueError("Cannot enable auxiliary task at test time")
        images = to_image_list(images)
        features = self.backbone(images.tensors, params=self.get_subdict(params, "backbone"))

        #if params is not None:
        #   print(self.get_subdict(params,"backbone").keys())

        if auxiliary_task and self.cfg.MODEL.SELF_SUPERVISOR.TYPE == "rotation":
            # self._log_image_tensorboard(images, targets, 5)
            straight_features = features
            rotated_img_features = {0: straight_features}

            rotated_images = {0: images}

            for rot_i in range(1, 4):
                rot_images = []
                for img, img_size in zip(images.tensors, images.image_sizes):
                    rot_image, rot_index = SelfSup_Scrambler.rotate_single(img, rot_i)
                    rot_images.append(rot_image)

                # need to move to gpu?
                stacked_tensor = torch.stack(rot_images)
                r_features = self.backbone(stacked_tensor, params=self.get_subdict(params, "backbone"))
                rotated_img_features[rot_i] = r_features
                rotated_images[rot_i] = to_image_list(rot_images)

        if not meta:
           if targets is not None or not self.training:

              proposals, proposal_losses = self.rpn(images, features, targets, params=self.get_subdict(params, "rpn"))
              if self.roi_heads:
                  x, result, detector_losses = self.roi_heads(features, proposals, targets, params=self.get_subdict(params, "roi_heads"))
              else:
                  # RPN-only models don't have roi_heads
                  x = features
                  result = proposals
                  detector_losses = {}

        losses = {}

        pseudo_targets = None

        if auxiliary_task and self.cfg.MODEL.SELF_SUPERVISOR.TYPE == "rotation":
            # we use the *result* (a boxlist) as a list of boxes to perform the self supervised task
            if self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "detections" or ((self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "targets") and targets is None):

                test_result = self.obtain_pseudo_labels(images, features)
                if meta and len(test_result[0]) == 0:
                    self.global_step += 1
                    losses["aux_loss"] = -1
                    return losses["aux_loss"]


            elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "targets":

                test_result = targets

            elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "images":

                image_sizes = images.image_sizes
                test_result = []

                for height, width in image_sizes:
                    xmin = 0
                    ymin = 0
                    xmax = width
                    ymax = height
                    bbox = torch.tensor([[xmin,ymin,xmax,ymax]], dtype=torch.float)
                    boxlist = BoxList(bbox, (width, height))

                    boxlist = boxlist.to(images.tensors.device)
                    test_result.append(boxlist)

            elif self.cfg.MODEL.SELF_SUPERVISOR.REGIONS == "crop":

                image_sizes = images.image_sizes

                test_result = []

                for height, width in image_sizes:
                    xmin, ymin, xmax, ymax = self.random_crop_image(width, height)

                    bbox = torch.tensor([[xmin,ymin,xmax,ymax]], dtype=torch.float)
                    boxlist = BoxList(bbox, (width, height))

                    boxlist = boxlist.to(images.tensors.device)

                    test_result.append(boxlist)

            rotated_regions = {0: test_result}
            for rot_i in range(1, 4):
                r_result = [res[::] for res in test_result]

                for idx, box_list in enumerate(r_result):
                    rotated_boxes = box_list.transpose(rot_i + 1)

                    r_result[idx] = rotated_boxes

                rotated_regions[rot_i] = r_result

            # log images
            #for rot_i in range(0, 4):
            #    self._log_image_tensorboard(rotated_images[rot_i], rotated_regions[rot_i], rot_i)

            pooling_res = []
            rot_target_batch = []
            for idx_in_batch in range(len(test_result)):
                mul = 1
                rot_target = torch.ones((len(test_result[idx_in_batch]) * mul), dtype=torch.long)
                for r in range(len(test_result[idx_in_batch])):
                    rot = random.randint(0,3)
                    features_r = rotated_img_features[rot]
                    regions_r = rotated_regions[rot][idx_in_batch][[r]]
                    l_regions_r = [regions_r]
                    pooled_features = self.region_feature_extractor(features_r, l_regions_r, params=self.get_subdict(params, "region_feature_extractor"))
                    pooled_features = self.ss_adaptive_pooling(pooled_features)
                    pooled_features = pooled_features.view(pooled_features.size(0), -1)
                    class_preds = self.ss_classifier(self.ss_dropout(pooled_features), params=self.get_subdict(params, "ss_classifier"))
                    pooling_res.append(class_preds)
                    rot_target[r] = rot
                rot_target_batch.append(rot_target)

            if len(pooling_res) > 0:
                pooling_res = torch.stack(pooling_res).squeeze(dim=1)
                rot_target_batch = torch.cat(rot_target_batch).to(pooling_res.device)
                aux_loss = self.ss_criterion(pooling_res, rot_target_batch)
                aux_loss = aux_loss.mean()
                # add to dictionary of losses
                losses["aux_loss"] = aux_loss


        if self.training:
               if meta:
                  self.global_step +=1
                  return losses["aux_loss"]
               elif targets is not None:
                   losses.update(detector_losses)
                   losses.update(proposal_losses)
               self.global_step += 1
               return losses


        return result
