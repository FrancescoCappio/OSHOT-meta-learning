# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import copy
import random 

import torch
import torch.optim as optim
from tqdm import tqdm
from apex import amp

from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from ..utils.log_image_bb import log_test_image
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, oshot_breakpoints, timer=None, cfg=None):
    results = [{} for _ in range(len(oshot_breakpoints))]

    checkpoint = copy.deepcopy(model.state_dict())
    cpu_device = torch.device("cpu")

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    
    logging_enabled = True
    if logging_enabled:
        import tensorboardX
        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        assert len(images) == 1, 'Must work with a batch size of 1 for OSHOT'
        # Load checkpoint
        model.load_state_dict(checkpoint)

        model.eval()
        with torch.no_grad():
            output = model(images.to(device))
            torch.cuda.synchronize()
            output = [o.to(cpu_device) for o in output]
        if logging_enabled:
            image_name = data_loader.dataset.get_img_name(image_ids[0])
            log_test_image(cfg, summary_writer, "detections_0_its".format(), image_ids[0], images, output, image_name=image_name)

        for oshot_it in range(cfg.MODEL.SELF_SUPERVISOR.OSHOT_ITERATIONS):
            model.train()
            optimizer.zero_grad()

            imgs = copy.deepcopy(images)
            # random horizontal flip:
            if random.random() < 0.5:
                imgs.tensors = torch.flip(imgs.tensors,dims=(3,))

            # apply forward 
            j_loss_dict = model(imgs.to(device), auxiliary_task=True)

            # compute total loss
            j_losses = sum(loss for loss in j_loss_dict.values())
            
            # apply self supervised weight
            losses_weights = [cfg.MODEL.SELF_SUPERVISOR.WEIGHT if 'aux' in k else 1 for k in j_loss_dict.keys()]
            j_losses = sum(loss * weight for loss, weight in zip(j_loss_dict.values(), losses_weights))

            if oshot_it < cfg.MODEL.SELF_SUPERVISOR.OSHOT_WARMUP:
                j_losses = j_losses * (oshot_it / cfg.MODEL.SELF_SUPERVISOR.OSHOT_WARMUP)
            try:
                j_losses.backward()

                optimizer.step()
            except AttributeError:
                print("AttributeError, maybe no detections detected?")
                pass
            
            if (oshot_it + 1) in oshot_breakpoints:
                model.eval()
                with torch.no_grad():
                    if timer:
                        timer.tic()
                    if cfg.TEST.BBOX_AUG.ENABLED:
                        output = im_detect_bbox_aug(model, images, device)
                    else:
                        output = model(images.to(device))
                    if timer:
                        if not cfg.MODEL.DEVICE == 'cpu':
                            torch.cuda.synchronize()
                        timer.toc()
                    output = [o.to(cpu_device) for o in output]
                results[oshot_breakpoints.index(oshot_it+1)].update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
                if logging_enabled:
                    log_test_image(cfg, summary_writer, "detections_{}_its".format(oshot_it+1), image_ids[0], images, output, image_name=image_name)

    return results


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        import pdb; pdb.set_trace() 
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def oshot_inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        oshot_breakpoints=(),
        cfg=None
):
    # add last breakpoint to oshot 
    oshot_breakpoints = (*oshot_breakpoints, cfg.MODEL.SELF_SUPERVISOR.OSHOT_ITERATIONS)
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, oshot_breakpoints, inference_timer, cfg)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    for i, p in enumerate(predictions):
        predictions[i] = _accumulate_predictions_from_multiple_gpus(p)
    if not is_main_process():
        return

    if output_folder:
        for i in range(len(predictions)):
            torch.save(predictions[i], os.path.join(output_folder, "oshot_predictions_%d.pth" % oshot_breakpoints[i]))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    evaluations = []
    for i,p in enumerate(predictions):
        evaluations.append(evaluate(dataset=dataset,
                           predictions=p,
                           output_folder=output_folder,
                           **extra_args))
    return evaluations
