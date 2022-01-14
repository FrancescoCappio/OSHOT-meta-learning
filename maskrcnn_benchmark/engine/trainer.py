# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.image_list import ImageList, to_image_list
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.data import make_data_loader, make_data_sampler, make_batch_data_sampler
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.data.collate_batch import BatchCollator

from apex import amp
from ..data.transforms.transforms import DeNormalize, ToPixelDomain
from ..data.transforms import build_transforms
import numpy as np


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def run_test(cfg, model, distributed=False, test_mode="test", val_sets_dict=None):
    synchronize()
    model.eval()
    if distributed:
        model_orig = model
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    if test_mode == "test":
        dataset_names = cfg.DATASETS.TEST
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    else:
        dataset_names = val_sets_dict.keys()
        data_loaders_val = []

        # create data loaders for validation datasets
        num_gpus = get_world_size()
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        shuffle = False if not distributed else True
        images_per_gpu = images_per_batch // num_gpus
        num_iters = None
        start_iter = 0
        aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

        val_transforms = None if cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, False)
        for k, ds in val_sets_dict.items():
            ds.set_keep_difficult(True)

            ds.set_transforms(val_transforms)
            sampler = make_data_sampler(ds, shuffle, distributed)
            batch_sampler = make_batch_data_sampler(
                ds, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
            )
            collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
            num_workers = cfg.DATALOADER.NUM_WORKERS
            data_loader = torch.utils.data.DataLoader(
                ds,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collator,
            )
            data_loaders_val.append(data_loader)

    sum_mAPs = 0
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        results = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            cfg=cfg,
        )
        synchronize()
        if distributed and not dist.get_rank() == 0:
            continue
        sum_mAPs += results["map"]
    if distributed:
        model = model_orig
    model.train()

    if test_mode == "val":
        train_transforms = build_transforms(cfg, True)
        for k, ds in val_sets_dict.items():
            ds.set_keep_difficult(False)
            ds.set_transforms(train_transforms)
    return sum_mAPs / len(dataset_names)


def _scale_back_image(cfg, img):
    orig_image = img.numpy()
    t1 = np.transpose(orig_image, (1, 2, 0))
    transform1 = DeNormalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    #transform2 = ToPixelDomain()
    orig_image = transform1(t1)
    orig_image = orig_image.astype(np.uint8)
    orig_image = np.transpose(orig_image, (2,0,1))
    
    # transpose from bgr to rgb
    return orig_image[::-1,:,:]


def _log_images_tensorboard(cfg, global_step, summary_writer, image_list, targets, j_images=None):
    from random import randrange
    size = len(image_list.image_sizes)
    i = randrange(size)
    imagei_tensor = image_list.tensors[i]
    imagei_tensor = imagei_tensor.cpu()
    imagei_size = image_list.image_sizes[i]

    #correct size
    imagei = imagei_tensor[:imagei_size[1], :imagei_size[0]]

    rescaled = _scale_back_image(cfg, imagei)
    summary_writer.add_image('img', rescaled, global_step=global_step)
    if j_images is not None:
        j_imagei_tensor = j_images.tensors[i]
        j_imagei_tensor = j_imagei_tensor.cpu()
        j_imagei_size = j_images.image_sizes[i]
        j_imagei = j_imagei_tensor[:imagei_size[1], :imagei_size[0]]
        rescaled2 = _scale_back_image(cfg, j_imagei)
        summary_writer.add_image('img_ss', rescaled2, global_step=global_step)


def do_train(
        cfg,
        args,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        val_sets_dict=None
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    if args.use_tensorboard:
        import tensorboardX
        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    max_iter = cfg.SOLVER.MAX_ITER
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    data_loader = iter(data_loader)

    if args.return_best:
        best_map = -1
        best_model = "model"

    for iteration in range(start_iter, max_iter):
        # TODO: does doing next(data_loader) consider that the data_loader should start from iteration start_iter? 
        # To be noticed that the batch sampler (inside the data loader) should know about iterations

        images, targets, _ = next(data_loader)

        if any(len(target) < 1 for target in targets):
            logger.error(
                f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        if not cfg.MODEL.SELF_SUPERVISED or (cfg.MODEL.SELF_SUPERVISED and not cfg.MODEL.SELF_SUPERVISOR.TYPE == "rotation"):
            loss_dict = model(images, targets)
        else:
            # self supervised rotation
            loss_dict = model(images, targets, auxiliary_task=True)

        losses_weights = [cfg.MODEL.SELF_SUPERVISOR.WEIGHT if 'aux' in k else 1 for k in loss_dict.keys()]
        losses = sum(loss * weight for loss, weight in zip(loss_dict.values(), losses_weights))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % args.log_step == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if args.use_tensorboard:
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=iteration)
                for loss_name, loss_item in meters.meters.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item.median, global_step=iteration)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=iteration)
                summary_writer.add_scalar('max_mem', torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

                _log_images_tensorboard(cfg, iteration, summary_writer, images, targets)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)

        if iteration % args.eval_step == 0 or iteration == max_iter:
            if args.eval_mode == "test":
                avg_map = run_test(cfg, model, args.distributed, args.eval_mode)
            else:
                avg_map = run_test(cfg, model, args.distributed, args.eval_mode, val_sets_dict)

            if avg_map is not None and args.use_tensorboard:
                summary_writer.add_scalar('eval_avg_map', avg_map, global_step=iteration)
            if avg_map is not None and args.return_best:
                if avg_map > best_map:
                    best_map = avg_map
                    best_model = "model_{:07d}".format(iteration)
                    logger.info(
                        "With iteration {} passed the best! New best avg map: {:4f}".format(iteration, best_map))
                    checkpointer.save("best_model", **arguments)
                else:
                    logger.info(
                        "With iteration {} the best has not been reached. Best avg map: {:4f}, Current avg mAP: {:4f}".format(
                            iteration, best_map, avg_map))

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if args.return_best:
        logger.info("The best model is '{}' with an average mAP: {}".format(best_model, best_map))
        checkpointer.load(cfg.OUTPUT_DIR + "/best_model.pth", use_latest=False)
