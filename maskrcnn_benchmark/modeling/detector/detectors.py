# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .meta_generalized_rcnn import meta_generalizedRCNN



_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "Meta_GeneralizedRCNN" : meta_generalizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
