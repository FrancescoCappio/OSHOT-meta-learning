import torch
import torch.nn as nn
import logging
import types

from .convert_meta import convert_to_meta
from .generalized_rcnn import GeneralizedRCNN

def meta_generalizedRCNN(config):
   model = GeneralizedRCNN(config)
   meta_model = convert_to_meta(model)
   return meta_model
