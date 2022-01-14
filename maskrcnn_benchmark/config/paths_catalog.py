# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/instances_train2014.json"
        },
        "coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_val2014.json"
        },
        "coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_minival2014.json"
        },
        "coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/instances_valminusminival2014.json"
        },
        "keypoints_coco_2014_train": {
            "img_dir": "coco/train2014",
            "ann_file": "coco/annotations/person_keypoints_train2014.json",
        },
        "keypoints_coco_2014_val": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_val2014.json"
        },
        "keypoints_coco_2014_minival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_minival2014.json",
        },
        "keypoints_coco_2014_valminusminival": {
            "img_dir": "coco/val2014",
            "ann_file": "coco/annotations/person_keypoints_valminusminival2014.json",
        },
        "voc_2007_train": {
            "data_dir": "VOC2007",
            "split": "train"
        },
        "voc_2007_trainval": {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_trainval_subset': {
            "data_dir": "VOC2007",
            "split": "trainval",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        "voc_2007_train_cocostyle": {
            "img_dir": "VOC2007/JPEGImages",
            "ann_file": "VOC2007/Annotations/pascal_train2007.json"
        },
        "voc_2007_val": {
            "data_dir": "VOC2007",
            "split": "val"
        },
        "voc_2007_val_cocostyle": {
            "img_dir": "VOC2007/JPEGImages",
            "ann_file": "VOC2007/Annotations/pascal_val2007.json"
        },
        "voc_2007_test": {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2007_test_subset': {
            "data_dir": "VOC2007",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        "voc_2007_test_cocostyle": {
            "img_dir": "VOC2007/JPEGImages",
            "ann_file": "VOC2007/Annotations/pascal_test2007.json"
        },
        "voc_2012_train": {
            "data_dir": "VOC2012",
            "split": "train"
        },
        "voc_2012_trainval": {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_trainval_subset': {
            "data_dir": "VOC2012",
            "split": "trainval",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        "voc_2012_train_cocostyle": {
            "img_dir": "VOC2012/JPEGImages",
            "ann_file": "VOC2012/Annotations/pascal_train2012.json"
        },
        "voc_2012_val": {
            "data_dir": "VOC2012",
            "split": "val"
        },
        "voc_2012_val_cocostyle": {
            "img_dir": "VOC2012/JPEGImages",
            "ann_file": "VOC2012/Annotations/pascal_val2012.json"
        },
        "amds_train": {
            "data_dir": "amds",
            "split": "train",
        },
        "amds_test": {
            "data_dir": "amds",
            "split": "test",
        },
        "voc_2012_test": {
            "data_dir": "VOC2012",
            "split": "test"
            # PASCAL VOC2012 doesn't made the test annotations available, so there's no json annotation
        },
        "cityscapes_detection_train": {
            "data_dir": "cityscapes",
            "split": "train"
        },
        "cityscapes_detection_val": {
            "data_dir": "cityscapes",
            "split": "val"
        },
        "cityscapes_detection_foggy_train": {
            "data_dir": "cityscapes",
            "split": "foggy_train"
        },
        "cityscapes_1_00_detection_foggy_train": {
            "data_dir": "cityscapes_1_00",
            "split": "foggy_train"
        },
        "cityscapes_1_01_detection_foggy_train": {
            "data_dir": "cityscapes_1_01",
            "split": "foggy_train"
        },
        "cityscapes_1_02_detection_foggy_train": {
            "data_dir": "cityscapes_1_02",
            "split": "foggy_train"
        },
        "cityscapes_detection_foggy_val": {
            "data_dir": "cityscapes",
            "split": "foggy_val"
        },
        "cityscapes_fine_instanceonly_seg_train_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
        },
        "cityscapes_fine_instanceonly_seg_val_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
        },
        "cityscapes_fine_instanceonly_seg_test_cocostyle": {
            "img_dir": "cityscapes/images",
            "ann_file": "cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
        },
        'clipart_train_subset': {
            "data_dir": "clipart",
            "split": "train",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'clipart_test_subset': {
            "data_dir": "clipart",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'clipart_test_train_subset': {
            "data_dir": "clipart",
            "split": ("test", "train"),
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'clipart_train': {
            "data_dir": "clipart",
            "split": "train"
        },
        'clipart_1_00_train': {
            "data_dir": "clipart_1_00",
            "split": "train"
        },
        'clipart_1_01_train': {
            "data_dir": "clipart_1_01",
            "split": "train"
        },
        'clipart_1_02_train': {
            "data_dir": "clipart_1_02",
            "split": "train"
        },
        'clipart_10_0_train': {
            "data_dir": "clipart_10_0",
            "split": "train"
        },
        'clipart_10_1_train': {
            "data_dir": "clipart_10_1",
            "split": "train"
        },
        'clipart_10_2_train': {
            "data_dir": "clipart_10_2",
            "split": "train"
        },
        'clipart_25_0_train': {
            "data_dir": "clipart_25_0",
            "split": "train"
        },
        'clipart_25_1_train': {
            "data_dir": "clipart_25_1",
            "split": "train"
        },
        'clipart_25_2_train': {
            "data_dir": "clipart_25_2",
            "split": "train"
        },
        'clipart_25_3_train': {
            "data_dir": "clipart_25_3",
            "split": "train"
        },
        'clipart_50_0_train': {
            "data_dir": "clipart_50_0",
            "split": "train"
        },
        'clipart_50_1_train': {
            "data_dir": "clipart_50_1",
            "split": "train"
        },
        'clipart_test': {
            "data_dir": "clipart",
            "split": "test"
        },
        'clipart_100_test': {
            "data_dir": "clipart_100",
            "split": "test"
        },
        'clipart_100_fake_test': {
            "data_dir": "clipart_100_fake",
            "split": "test"
        },
        'bicycle_test': {
            "data_dir": "bicycle_on_twitter",
            "split": "test"
        },
        'bicycle_fake_test': {
            "data_dir": "bicycle_fake",
            "split": "test"
        },
        'bicycle_social_test': {
            "data_dir": "bicycle_social",
            "split": "test"
        },
        'social_bikes_test': {
            "data_dir": "social_bikes",
            "split": "test"
        },
        'bicycle_all_test': {
            "data_dir": "bicycle_all",
            "split": "test"
        },
        'bicycle_all_fake_test': {
            "data_dir": "bicycle_all_fake",
            "split": "test"
        },
        'clipart_test_train': {
            "data_dir": "clipart",
            "split": ("test", "train")
        },
        'watercolor_train_subset': {
            "data_dir": "watercolor",
            "split": "train",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'watercolor_test_subset': {
            "data_dir": "watercolor",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'watercolor_test_train_subset': {
            "data_dir": "watercolor",
            "split": "instance_level_annotated",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'watercolor_test': {
            "data_dir": "watercolor",
            "split": "test"
        },
        'watercolor_train': {
            "data_dir": "watercolor",
            "split": "train"
        },
        'watercolor_1_00_train': {
            "data_dir": "watercolor_1_00",
            "split": "train"
        },
        'watercolor_1_01_train': {
            "data_dir": "watercolor_1_01",
            "split": "train"
        },
        'watercolor_1_02_train': {
            "data_dir": "watercolor_1_02",
            "split": "train"
        },
        'watercolor_10_0_train': {
            "data_dir": "watercolor_10_0",
            "split": "train"
        },
        'watercolor_10_1_train': {
            "data_dir": "watercolor_10_1",
            "split": "train"
        },
        'watercolor_10_2_train': {
            "data_dir": "watercolor_10_2",
            "split": "train"
        },
        'watercolor_25_0_train': {
            "data_dir": "watercolor_25_0",
            "split": "train"
        },
        'watercolor_25_1_train': {
            "data_dir": "watercolor_25_1",
            "split": "train"
        },
        'watercolor_25_2_train': {
            "data_dir": "watercolor_25_2",
            "split": "train"
        },
        'watercolor_25_3_train': {
            "data_dir": "watercolor_25_3",
            "split": "train"
        },
        'watercolor_50_0_train': {
            "data_dir": "watercolor_50_0",
            "split": "train"
        },
        'watercolor_50_1_train': {
            "data_dir": "watercolor_50_1",
            "split": "train"
        },
        'comic_train_subset': {
            "data_dir": "comic",
            "split": "train",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'comic_test_subset': {
            "data_dir": "comic",
            "split": "test",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'comic_test_train_subset': {
            "data_dir": "comic",
            "split": "instance_level_annotated",
            "desired_classes_subset": ('bicycle', 'bird', 'car', 'cat', 'dog', 'person')
        },
        'comic_test': {
            "data_dir": "comic",
            "split": "test"
        },
        'comic_train': {
            "data_dir": "comic",
            "split": "train"
        },
        'comic_1_00_train': {
            "data_dir": "comic_1_00",
            "split": "train"
        },
        'comic_1_01_train': {
            "data_dir": "comic_1_01",
            "split": "train"
        },
        'comic_1_02_train': {
            "data_dir": "comic_1_02",
            "split": "train"
        },
        'comic_10_0_train': {
            "data_dir": "comic_10_0",
            "split": "train"
        },
        'comic_10_1_train': {
            "data_dir": "comic_10_1",
            "split": "train"
        },
        'comic_10_2_train': {
            "data_dir": "comic_10_2",
            "split": "train"
        },
        'comic_25_0_train': {
            "data_dir": "comic_25_0",
            "split": "train"
        },
        'comic_25_1_train': {
            "data_dir": "comic_25_1",
            "split": "train"
        },
        'comic_25_2_train': {
            "data_dir": "comic_25_2",
            "split": "train"
        },
        'comic_25_3_train': {
            "data_dir": "comic_25_3",
            "split": "train"
        },
        'comic_50_0_train': {
            "data_dir": "comic_50_0",
            "split": "train"
        },
        'comic_50_1_train': {
            "data_dir": "comic_50_1",
            "split": "train"
        },
        'sim10k_subset': {
            "data_dir": "Sim10k",
            "split": "trainval10k",
            "desired_classes_subset": ('car',)
        },
        'cityscapes_sim': {
            "data_dir": "cityscapes",
            "split": "val"
        },
        'kitti_trainval': {
            "data_dir": "kitti/VOC2012",
            "split": "trainval"
        }
    }

    @staticmethod
    def get(name):
        if "subset" in name:
            root_dir = DatasetCatalog.DATA_DIR

            if "voc_2007" in name:
                if 'VOC07_ROOT' in os.environ:
                    root_dir = os.environ['VOC07_ROOT']
            elif "voc_2012" in name:
                if 'VOC12_ROOT' in os.environ:
                    root_dir = os.environ['VOC12_ROOT']
            elif "clipart" in name:
                if 'CLIPART_ROOT' in os.environ:
                    root_dir = os.environ['CLIPART_ROOT']
            elif "watercolor" in name:
                if 'WATERCOLOR_ROOT' in os.environ:
                    root_dir = os.environ['WATERCOLOR_ROOT']
            elif "comic" in name:
                if 'COMIC_ROOT' in os.environ:
                    root_dir = os.environ['COMIC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root_dir, attrs["data_dir"]),
                split=attrs["split"],
                desired_classes_subset=attrs["desired_classes_subset"]
            )
            return dict(
                factory="PascalVOCModelDataset",
                args=args)
        elif "clipart" in name or "comic" in name or "watercolor" in name or "amds" in name or "bicycle" in name or "social_bikes" in name:
            root_dir = DatasetCatalog.DATA_DIR
            if 'CLIPART_ROOT' in os.environ:
                    root_dir = os.environ['CLIPART_ROOT']
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(root_dir, attrs["data_dir"]),
                split=attrs["split"]
            )
            return dict(
                factory="PascalVOCModelDataset",
                args=args)
        elif "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        elif "kitti" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="KITTIVOCModelDataset",
                args=args,
            )
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            if "sim" not in name:
                return dict(
                    factory="CityscapesDataset",
                    args=args,
                )
            else:
                return dict(
                    factory="CityscapesSimDataset",
                    args=args,
                )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
