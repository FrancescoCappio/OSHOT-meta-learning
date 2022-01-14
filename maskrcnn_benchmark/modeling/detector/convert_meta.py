import torch.nn as nn
import torch
import torch.nn.functional as F
from .grcnn_forward import grcnn_forward, meta_obtain_pseudo_labels
from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaLinear, MetaBatchNorm2d
from torchvision.models import resnet
import types
from maskrcnn_benchmark.modeling.roi_heads.roi_heads import CombinedROIHeads
from maskrcnn_benchmark.modeling.roi_heads.box_head.box_head import ROIBoxHead
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_predictors import FastRCNNPredictor
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling.rpn.rpn import RPNModule, RPNHead
from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.modeling.backbone.resnet import Bottleneck, BaseStem, ResNet, ResNetHead
from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN


def bottleneck_forward(self, x, params=None):
    # if params is not None:
    #   print("PARAAMETRI BOTTLENECK: {}".format(params.keys()))
    #   print("PARAMETRI DOWNSAMPLE: {}".format(self.get_subdict(params, "downsample.0").keys()))

    identity = x

    out = self.conv1(x, params=self.get_subdict(params, "conv1"))
    out = self.bn1(out)
    out = F.relu_(out)

    out = self.conv2(out, params=self.get_subdict(params, "conv2"))
    out = self.bn2(out)
    out = F.relu_(out)

    out = self.conv3(out, params=self.get_subdict(params, "conv3"))
    out = self.bn3(out)

    if self.downsample is not None:
       identity = self.downsample(x, params=self.get_subdict(params, "downsample"))

    out += identity
    out = F.relu_(out)

    return out


def base_stem_forward(self, x, params=None):
    #if params is not None:
    #   print("PARAMETRI STEM:{}".format(params.keys()))

    x = self.conv1(x, params=self.get_subdict(params, "conv1"))
    x = self.bn1(x)
    x = F.relu_(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    return x

def resnet_forward(self, x, params=None):
    outputs = []
    x = self.stem(x, params = self.get_subdict(params, "stem"))
    for stage_name in self.stages:
        # print("STAGE_NAME:{}".format(stage_name))
        x = getattr(self, stage_name)(x, params=self.get_subdict(params, stage_name))
        if self.return_features[stage_name]:
           outputs.append(x)
    return outputs

def rpn_module_forward(self, images, features, targets=None, params=None):
    objectness, rpn_box_regression = self.head(features, params=self.get_subdict(params, "head"))
    anchors = self.anchor_generator(images, features)

    if self.training:
       return self._forward_train(anchors, objectness, rpn_box_regression, targets)
    else:
       return self._forward_test(anchors, objectness, rpn_box_regression)

def rpn_head_forward(self, x, params=None):
    logits = []
    bbox_reg = []
    for feature in x:
        t = F.relu(self.conv(feature, params=self.get_subdict(params, "conv")))
        logits.append(self.cls_logits(t, params=self.get_subdict(params, "cls_logits")))
        bbox_reg.append(self.bbox_pred(t, params=self.get_subdict(params, "bbox_pred")))
    return logits, bbox_reg

def resnet50conv5roi_forward(self, x, proposals, params=None):
    x = self.pooler(x, proposals)
    x = self.head(x, params=self.get_subdict(params, "head"))
    return x

def resnethead_forward(self, x, params=None):
    for stage in self.stages:
        x = getattr(self, stage)(x, params=self.get_subdict(params, stage))
    return x

def fastrcnnpredictor_forward(self, x, params=None):
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    cls_logit = self.cls_score(x, params=self.get_subdict(params, "cls_score"))
    bbox_pred = self.bbox_pred(x, params=self.get_subdict(params, "bbox_pred"))
    return cls_logit, bbox_pred

def roiboxhead_forward(self, features, proposals, targets=None, params=None):
    if self.training:
       with torch.no_grad():
           proposals = self.loss_evaluator.subsample(proposals, targets)

    x = self.feature_extractor(features, proposals, params=self.get_subdict(params, "feature_extractor"))
    class_logits, box_regression = self.predictor(x, params=self.get_subdict(params, "predictor"))

    if not self.training:
        result = self.post_processor((class_logits, box_regression), proposals)
        return x, result, {}

    loss_classifier, loss_box_reg = self.loss_evaluator([class_logits],[box_regression]) 
    return (
        x,
        proposals,
        dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
    )

def combinedroiheads_forward(self, features, proposals, targets=None, params=None):
    losses = {}
    x, detections, loss_box = self.box(features, proposals, targets, params=self.get_subdict(params, "box"))
    losses.update(loss_box)
    return x, detections, losses

#inside this function is not considered the Batch normalization layer, because our G-RCNN use the freezed version of BatchNormalization
def convert_modules(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
           mod = getattr(model, child_name)
           newmod = MetaConv2d(mod.in_channels, mod.out_channels, kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding, bias=not mod.bias is None, groups=mod.groups)
           newmod.load_state_dict(mod.state_dict())
           setattr(model, child_name, newmod)
        elif isinstance(child, nn.Linear):
             mod = getattr(model, child_name)
             newmod = MetaLinear(mod.in_features, mod.out_features, bias=not mod.bias is None)
             newmod.load_state_dict(mod.state_dict())
             setattr(model, child_name, newmod)
        elif isinstance(child, FrozenBatchNorm2d):
             continue
        elif isinstance(child, nn.BatchNorm2d):
             mod = getattr(model, child_name)
             newmod = MetaBatchNorm2d(mod.num_features, eps=mod.eps, momentum=mod.momentum, affine=mod.affine, track_running_stats=mod.track_running_stats)
             newmod.load_state_dict(mod.state_dict())
             setattr(model, child_name, newmod)
        else:
            if isinstance(child, nn.Sequential):
               modules = [mod for mod in child]
               newmod = MetaSequential(*modules)
               setattr(model, child_name, newmod)
               convert_modules(newmod)
            elif isinstance(child, Bottleneck):
               newmod = MetaModule()
               for k,v in child.__dict__.items():
                   setattr(newmod, k, v)

               newmod.forward = types.MethodType(bottleneck_forward, newmod)
               setattr(model, child_name, newmod)
               convert_modules(newmod)
            elif isinstance(child, BaseStem):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(base_stem_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, ResNet):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(resnet_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, RPNModule):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(rpn_module_forward, newmod)
                 newmod._forward_train = child._forward_train
                 newmod._forward_test = child._forward_test
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, RPNHead):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(rpn_head_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, ResNet50Conv5ROIFeatureExtractor):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(resnet50conv5roi_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, ResNetHead):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(resnethead_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, FastRCNNPredictor):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(fastrcnnpredictor_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, ROIBoxHead):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(roiboxhead_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            elif isinstance(child, CombinedROIHeads):
                 newmod = MetaModule()
                 for k,v in child.__dict__.items():
                     setattr(newmod, k, v)

                 newmod.forward = types.MethodType(combinedroiheads_forward, newmod)
                 setattr(model, child_name, newmod)
                 convert_modules(newmod)
            else:

                 convert_modules(child)


def convert_to_meta(net):

    G_RCNN = False
    if isinstance(net, GeneralizedRCNN):
       G_RCNN = True

    convert_modules(net)
    meta_net = MetaModule()
    for k,v in net.__dict__.items():
        setattr(meta_net, k, v)

    if G_RCNN:
       meta_net.forward = types.MethodType(grcnn_forward, meta_net)
       for key, value in meta_net.named_parameters():
           if 'stem' in key or 'layer1' in key:
               value.requires_grad = False
    else:
       meta_net.forward = net.forward

    if(G_RCNN):
      meta_net._scale_back_image = net._scale_back_image
      meta_net._log_image_tensorboard = net._log_image_tensorboard
      meta_net._compute_colors_for_labels = net._compute_colors_for_labels
      meta_net._overlay_boxes = net._overlay_boxes
      meta_net.random_crop = net.random_crop
      meta_net.random_crop_image = net.random_crop_image
      meta_net.obtain_pseudo_labels = net.obtain_pseudo_labels
      #types.MethodType(meta_obtain_pseudo_labels, meta_net)

    print(meta_net)


    return meta_net
