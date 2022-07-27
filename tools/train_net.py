# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.trainer_baseline_transf import do_train as do_train_transf
from maskrcnn_benchmark.engine.trainer_full_meta_oshot import do_train_meta as do_train_full_meta_oshot
from maskrcnn_benchmark.engine.trainer_meta_oshot import do_train_meta as do_train_meta_oshot
from maskrcnn_benchmark.engine.trainer_oshot_transf import do_train_meta as do_train_oshot_transf
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, args, logger):
    local_rank = args.local_rank
    distributed = args.distributed

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    if args.eval_mode == "test":
        data_loaders = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
    else: # val
        data_loaders, val_sets_dict = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
            train_val_split=True
        )
    
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if args.meta:
        # FULL-OSHOT config 
        if args.meta_mode == "full_meta_oshot":
            if args.eval_mode == "test":
                logger.info("FULL META OSHOT eval_mode = TEST")
                do_train_full_meta_oshot(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments)
            else:
                logger.info("FULL META OSHOT eval_mode = VAL")
                do_train_full_meta_oshot(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments,val_sets_dict=val_sets_dict)
        # Meta-OSHOT config
        elif args.meta_mode == "meta_oshot":
            if args.eval_mode == "test":
                logger.info("META OSHOT eval_mode = TEST")
                do_train_meta_oshot(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments)
            else:
                logger.info("META OSHOT eval_mode = VAL")
                do_train_meta_oshot(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments,val_sets_dict=val_sets_dict)
        # Tran-OSHOT config
        elif args.meta_mode == "oshot_transf":
            if args.eval_mode == "test":
                logger.info("OSHOT TRANSF eval_mode = TEST")
                do_train_oshot_transf(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments)
            else:
                logger.info("OSHOT TRANSF eval_mode = VAL")
                do_train_oshot_transf(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments,val_sets_dict=val_sets_dict)
        else: 
            raise NotImplementedError(f"Meta mode {args.meta_mode} not known")
    else:
        # Tran-Baseline config 
        if args.enable_transf:
            if args.eval_mode == "test":
                logger.info("Baseline TRANSF eval_mode = TEST")
                do_train_transf(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments)
            else:
                logger.info("Baseline TRANSF eval_mode = VAL")
                do_train_transf(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments,val_sets_dict=val_sets_dict)
        # OSHOT/Baseline config
        else:
            if args.eval_mode == "test":
                do_train(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments)
            else:
                do_train(cfg,args,model,data_loaders,optimizer,scheduler,checkpointer,device,checkpoint_period,arguments,val_sets_dict=val_sets_dict)

    return model


def run_test(cfg, model, distributed):
    if distributed:
        print("Distributed!")
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            cfg=cfg,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--meta_pretraining_steps", type=int, default=10000, help="Number of meta learning iterations to perform at the end of pretraining training")
    parser.add_argument("--meta_iters", type=int, default=5, help="Number of meta learning iterations (inner loop steps) to perform on each sample in the pretraining phase")
    parser.add_argument("--meta", action='store_true', help="Enable meta learning in the last meta_pretraining_steps iterations")
    parser.add_argument("--enable_transf", action='store_true', help="Enable baseline training with transforms in the last meta_pretraining_steps")
    parser.add_argument("--meta_mode", type=str, help="Choose meta training strategy", choices=['full_meta_oshot', 'meta_oshot', 'oshot_transf'], default="full_meta_oshot")
    parser.add_argument(
        "--use_tensorboard",
        default=False,
        type=bool,
        help="Enable/disable tensorboard logging (disabled by default)"
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--log_step",
        default=50,
        type=int,
        help='Number of iteration for each log'
    )
    parser.add_argument(
        "--eval_mode",
        default="test",
        type=str,
        help='Use defined test datasets for periodic evaluation or use a validation split. Default: "test", alternative "val"'
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=5000,
        help="Number of iterations for periodic evaluation"
    )
    parser.add_argument(
        "--return_best",
        action='store_true',
        help=' Tests on best model instead of last model'
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()

