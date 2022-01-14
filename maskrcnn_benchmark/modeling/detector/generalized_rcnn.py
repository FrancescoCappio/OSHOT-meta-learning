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

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.global_step = 0

        self.region_feature_extractor = make_roi_box_feature_extractor(cfg, self.backbone.out_channels)

        self.cfg = cfg
        self.summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)

        self.ss_criterion = nn.CrossEntropyLoss(reduction='none')
        self.ss_adaptive_pooling = nn.AdaptiveAvgPool2d(1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        if cfg.MODEL.SELF_SUPERVISED and cfg.MODEL.SELF_SUPERVISOR.TYPE == "rotation":
            self.ss_dropout = nn.Dropout(p=cfg.MODEL.SELF_SUPERVISOR.DROPOUT) 
            self.ss_classifier = nn.Linear(self.region_feature_extractor.out_channels, 4)

    def _scale_back_image(self, img):
        orig_image = img.numpy()
        t1 = np.transpose(orig_image, (1, 2, 0))
        transform1 = DeNormalize(self.cfg.INPUT.PIXEL_MEAN, self.cfg.INPUT.PIXEL_STD)
        orig_image = transform1(t1)
        orig_image = orig_image.astype(np.uint8)
        orig_image = np.transpose(orig_image, (2, 0, 1))

        return orig_image[::-1,:,:]

    def _log_image_tensorboard(self, image_list, targets, rotation):

        image_tensor = image_list.tensors[0].cpu()
        image_size = image_list.image_sizes[0]

        targets = targets[0]

        # fix size
        image = image_tensor[:image_size[1], :image_size[0]]

        image = self._scale_back_image(image)

        result = image.copy()

        result = self._overlay_boxes(result, targets)

        self.summary_writer.add_image('img_{}'.format(rotation), np.transpose(result,(2,0,1)), global_step=self.global_step)

    def _compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        if not labels.dtype == torch.int64:
            self.palette = self.palette.float()
        colors = labels[:, None] * self.palette.to(labels.device)
        colors = (colors % 255).cpu().numpy().astype("uint8")
        return colors

    def _overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        if predictions.has_field("labels"):
            labels = predictions.get_field("labels")
        else:
            labels = torch.ones(len(predictions.bbox), dtype=torch.float)
        boxes = predictions.bbox

        colors = self._compute_colors_for_labels(labels).tolist()

        pil_image = ToPILImage()(np.transpose(image, (1,2,0)))
        draw = Draw(pil_image)
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

            draw.rectangle([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], outline=tuple(color), width=2)

        del draw
        return np.array(pil_image)


    def random_crop(self, feature, crop_size):
        w, h = feature.size()[2:]
        th, tw = crop_size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        crop = feature[:,:,j:j+tw,i:i+th]
        return crop

    def random_crop_image(self, image_width, image_height, crop_size=237):

        crop_size = min(image_width, image_height, crop_size)

        xmin = random.randint(0, image_width - crop_size)
        ymin = random.randint(0, image_height -crop_size)

        xmax = xmin + crop_size
        ymax = ymin + crop_size
        return xmin, ymin, xmax, ymax        

    def obtain_pseudo_labels(self, images, features):
        self.eval()

        test_proposals, _ = self.rpn(images, features)
        _, pseudo_targets, _ = self.roi_heads(features, test_proposals)

        #if len(pseudo_targets[0])==0:
        #    print("No pseudo targets!")
        self.train()
        return pseudo_targets

    def forward(self, images, targets=None, auxiliary_task=False):
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
        features = self.backbone(images.tensors)

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
                r_features = self.backbone(stacked_tensor)
                rotated_img_features[rot_i] = r_features
                rotated_images[rot_i] = to_image_list(rot_images)

        if targets is not None or not self.training:

            proposals, proposal_losses = self.rpn(images, features, targets)
            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets)
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
                    pooled_features = self.region_feature_extractor(features_r, l_regions_r)
                    pooled_features = self.ss_adaptive_pooling(pooled_features)
                    pooled_features = pooled_features.view(pooled_features.size(0), -1)
                    class_preds = self.ss_classifier(self.ss_dropout(pooled_features))
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
            if targets is not None:
                losses.update(detector_losses)
                losses.update(proposal_losses)
            self.global_step += 1
            return losses

        return result
