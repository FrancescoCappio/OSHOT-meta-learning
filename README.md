# Demo implementation of *Self-Supervision & Meta-Learning for One-Shot Unsupervised Cross-Domain Detection*

Link to the paper: https://arxiv.org/abs/2106.03496


This paper is an extension of our ECCV20: [One-Shot Unsupervised Cross-Domain Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610715.pdf), the code
is therefore based on the original implementation: https://github.com/VeloDC/oshot_detection.

The detection framework is inherited from [Maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and uses Pytorch and CUDA.

This readme will guide you through a full run of our method for the Pascal VOC -> AMD benchmarks. 

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

