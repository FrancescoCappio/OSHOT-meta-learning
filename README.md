# Demo implementation of *Self-Supervision & Meta-Learning for One-Shot Unsupervised Cross-Domain Detection*

Link to the paper: https://arxiv.org/abs/2106.03496


This paper is an extension of our ECCV20: [One-Shot Unsupervised Cross-Domain Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610715.pdf), the code
is therefore based on the original implementation: https://github.com/VeloDC/oshot_detection.

The detection framework is inherited from [Maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and uses Pytorch and CUDA.

This readme will guide you through a full run of our method for the Pascal VOC -> AMD benchmarks. 

## Implementation details 

We build on top of Faster-RCNN with a ResNet-50 Backbone pre-trained on ImageNet, 300 top proposals
after non-maximum-suppression, anchors at three scales (128, 256, 512) and three aspect ratios (1:1,
1:2, 2:1).

For OSHOT we train the base network for 70k iterations using SGD with momentum set at 0.9, the
initial learning rate is 0.001 and decays after 50k iterations. We use a batch size of 1, keep 
batch normalization layers fixed for both pretraining and adaptation phases and freeze the first 
2 blocks of ResNet50. The weight of the rotation task is set to λ=0.05.

FULL-OSHOT is actually trained in two steps. For the first 60k iterations the training is identical 
to that of OSHOT, while in the last 10k iterations the meta-learning procedure is activated. The 
inner loop optimization on the self-supervised task runs with η=5 iterations and the batch size 
is 2 to accomodate for two transformations of the original image. 
Specifically we used gray-scale and color-jitter with brightness, contrast, saturation and hue all set to 0.4. 
All the other hyperparameters remain unchanged as in OSHOT.

*Tran*-OSHOT differs from OSHOT only for the last 10k learning iterations, where the batch size is 2 
and the network sees multiple images with different visual appearance in one iteration. 
*Meta*-OSHOT is instead identical to FULL-OSHOT, made exception for the transformations which are dropped, 
thus the batch size is 1 also in the last 10k pretraining iterations. 

The adaptation phase is the same for all the variants: the model obtained from the pretraining phase is 
updated via fine-tuning of the self-supervised task. The batch size is equal to 1 and a dropout with 
probability p = 0.5 is added before the rotation classifier to prevent overfitting. The weight of the 
auxiliary task is increased to λ=0.2 to speed up the adaptation process. All the other hyperparameters 
and settings are the same used during the pretraining.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Datasets

Create a folder named `datasets` and include VOC2007 and VOC2012 source datasets (download from
[Pascal VOC's website](http://host.robots.ox.ac.uk/pascal/VOC/)).

Download and extract clipart1k, comic2k and watercolor2k from [authors'
website](https://naoto0804.github.io/cross_domain_detection/).

## Performing pretraining 

To perform a *standard* OSHOT pretraing using Pascal VOC as source dataset:

```bash
python tools/train_net.py --config-file configs/amd/voc_pretrain.yaml
```

To perform an improved pretraining using our meta-learning based procedure:

```bash
python tools/train_net.py --config-file configs/amd/voc_pretrain_meta.yaml --meta
```

Once you have performed a pretraining you can test the output model directly on the target domain or
perform the one-shot adaptation.

## Testing pretrained model

You can test a pretrained model on one of the AMD referring to the correct config-file. For example
for clipart:

```bash
python tools/test_net.py --config-file configs/amd/oshot_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```

## Performing the one-shot adaptation

To use OSHOT adaptation procedure and obtain results on one of the AMD please refer to one of the
config files. For example for clipart:

```bash
python tools/oshot_net.py --config-file configs/amd/oshot_clipart_target.yaml --ckpt <pretrain_output_dir>/model_final.pth
```

To perform the one-shot adaptation on a model trained with meta learning you need to refer the
corresponding `_meta` config file: 

```bash
python tools/oshot_net.py --config-file configs/amd/oshot_clipart_target_meta.yaml --ckpt <meta_pretrain_output_dir>/model_final.pth
```
# Qualitative results

Some visualizations for OSHOT, Full-OSHOT and baseline methods:

![Qualitative results](media/qualitative_results.png){width=70%}
