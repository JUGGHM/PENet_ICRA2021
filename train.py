import argparse
import typing
import logging
import os
import time
import torch

# Use GPU only if GPU is available and memory is more than 16GB
_MIN_TRAINING_MEMORY = 11e9
if (
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory > _MIN_TRAINING_MEMORY
):
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from penet import helper
from penet import models
from penet import criteria
from penet.metrics import AverageMeter, Result
from penet.dataloaders.kitti_loader import input_options, KittiDepth

_MULTI_BATCH_SIZE = 1
_RUNS_DIR_PATH = (
    Path(__file__).parent / "runs"
)  # This is where tensorboard summaries are saved
_DEFAULT_RESULTS_FOLDER = Path(__file__).parent / "results"


def _get_model_and_logger(options: argparse.Namespace):
    # Instantiate the model object
    if options.network_model == "e":
        model = models.ENet(options).to(device)
    else:
        if options.dilation_rate == 1:
            model = models.PENet_C1(options).to(device)
        elif options.dilation_rate == 2:
            model = models.PENet_C2(options).to(device)
        elif options.dilation_rate == 4:
            model = models.PENet_C4(options).to(device)

    # Instantiate logger
    logger = helper.logger(options)

    # Load training checkpoint if available
    if options.resume is not None:
        try:
            checkpoint = torch.load(str(options.resume), map_location=device)
        except ModuleNotFoundError:
            # Pre-trained models on GitHub fail after the refactoring because
            # metrics.py is not found by pickle so we use this hack
            import sys

            sys.path.insert(0, os.path.dirname(__file__))
            checkpoint = torch.load(str(options.resume), map_location=device)
        # Training checkpoint should be a dict
        assert isinstance(checkpoint, dict)
        if options.freeze_backbone:
            model.backbone.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"], strict=False)
        # Initialize best result in logger
        logger.best_result = checkpoint["best_result"]

    return model, logger


def _get_kitti_dataloaders(
    options: argparse.Namespace,
) -> typing.Tuple[DataLoader, DataLoader]:
    """Return data loaders configured according to cli options
    :param options: the cli options
    :return: a tuple with the training and validation data loaders
    """
    train_dataset = KittiDepth("train", options)
    train_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.workers,
        pin_memory=True,
        sampler=None,
    )
    val_dataset = KittiDepth("val", options)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # We set batch size 1 for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader


def iterate(mode, options, loader, model, optimizer, depth_criterion, logger, epoch):
    assert mode in ["train", "val"]

    actual_epoch = epoch - options.start_epoch + options.start_epoch_bias
    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    if mode == "train":
        for param in model.parameters():
            param.requires_grad = True
        if options.freeze_backbone is True:
            for param in model.module.backbone.parameters():
                param.requires_grad = False
        if options.network_model == "pe":
            model.module.encoder3.requires_grad = False
            model.module.encoder5.requires_grad = False
            model.module.encoder7.requires_grad = False
        model.train()
        lr = helper.adjust_learning_rate(options.lr, optimizer, actual_epoch, options)
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    for i, batch_data in enumerate(loader):
        dstart = time.time()
        batch_data = {
            key: val.to(device) for key, val in batch_data.items() if val is not None
        }

        gt = batch_data["gt"]

        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        if options.network_model == "e":
            start = time.time()
            st1_pred, st2_pred, pred = model(batch_data)
        else:
            start = time.time()
            pred = model(batch_data)

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        # inter loss_param
        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0
        round1, round2, round3 = 1, 3, None
        if actual_epoch <= round1:
            w_st1, w_st2 = 0.2, 0.2
        elif actual_epoch <= round2:
            w_st1, w_st2 = 0.05, 0.05
        else:
            w_st1, w_st2 = 0, 0

        if mode == "train":
            depth_loss = depth_criterion(pred, gt)

            if options.network_model == "e":
                st1_loss = depth_criterion(st1_pred, gt)
                st2_loss = depth_criterion(st2_pred, gt)
                loss = (
                    (1 - w_st1 - w_st2) * depth_loss
                    + w_st1 * st1_loss
                    + w_st2 * st2_loss
                )
            else:
                loss = depth_loss

            if i % _MULTI_BATCH_SIZE == 0:
                optimizer.zero_grad()
            loss.backward()

            if i % _MULTI_BATCH_SIZE == (_MULTI_BATCH_SIZE - 1) or i == (len(loader) - 1):
                optimizer.step()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))

        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            result.evaluate(pred.data, gt.data, photometric_loss)
            for meter in meters:
                meter.update(result, gpu_time, data_time, mini_batch_size)

            if mode == "val":
                logger.conditional_print(
                    mode, i, epoch, lr, len(loader), block_average_meter, average_meter
                )

            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and mode == "val":
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)
    return avg, is_best


class PENetSummaryWriter(SummaryWriter):
    def __init__(self, log_dir: str):
        super().__init__(log_dir=log_dir)

    def add_result(self, epoch: int, result: Result):
        self.add_scalars(
            "results",
            {
                "epoch": epoch,
                "rmse": result.rmse,
                "photo": result.photometric,
                "mae": result.mae,
                "irmse": result.irmse,
                "imae": result.imae,
                "mse": result.mse,
                "silog": result.silog,
                "squared_rel": result.squared_rel,
                "absrel": result.absrel,
                "lg10": result.lg10,
                "delta1": result.delta1,
                "delta2": result.delta2,
                "delta3": result.delta3,
            },
            epoch,
        )


def train(options: argparse.Namespace):
    # Instantiate model and logger
    model, logger = _get_model_and_logger(options)

    # Configure model and optimizer for training
    if options.freeze_backbone is True:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            model_named_params,
            lr=options.lr,
            weight_decay=options.weight_decay,
            betas=(0.9, 0.99),
        )
    elif options.network_model == "pe":
        model_bone_params = [
            p for _, p in model.backbone.named_parameters() if p.requires_grad
        ]
        model_new_params = [p for _, p in model.named_parameters() if p.requires_grad]
        model_new_params = list(set(model_new_params) - set(model_bone_params))
        optimizer = torch.optim.Adam(
            [
                {"params": model_bone_params, "lr": options.lr / 10},
                {"params": model_new_params},
            ],
            lr=options.lr,
            weight_decay=options.weight_decay,
            betas=(0.9, 0.99),
        )
    else:
        model_named_params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            model_named_params,
            lr=options.lr,
            weight_decay=options.weight_decay,
            betas=(0.9, 0.99),
        )

    model = torch.nn.DataParallel(model)

    # Get training and validation kitti loaders
    train_loader, val_loader = _get_kitti_dataloaders(options)

    # Instantiate loss
    depth_criterion = (
        criteria.MaskedMSELoss()
        if (options.criterion == "l2")
        else criteria.MaskedL1Loss()
    )

    # Configure tensorboard summary writers
    train_summary_dir_path = _RUNS_DIR_PATH / options.session_name / "training"
    train_summary_dir_path.mkdir(parents=True, exist_ok=True)
    train_writer = PENetSummaryWriter(str(train_summary_dir_path))
    val_summary_dir_path = _RUNS_DIR_PATH / options.session_name / "validation"
    val_summary_dir_path.mkdir(parents=True, exist_ok=True)
    val_writer = PENetSummaryWriter(str(val_summary_dir_path))

    for epoch in range(options.start_epoch, options.epochs):
        print("=> starting training epoch {} ..".format(epoch))

        # Train for one epoch and report results on summary
        result, _ = iterate(
            "train",
            options,
            train_loader,
            model,
            optimizer,
            depth_criterion,
            logger,
            epoch,
        )
        train_writer.add_result(epoch, result)

        # Evaluate on validation set and report results summary
        result, is_best = iterate(
            "val",
            options,
            val_loader,
            model,
            None,
            depth_criterion,
            logger,
            epoch,
        )
        val_writer.add_result(epoch, result)

        # Save checkpoint
        helper.save_checkpoint(
            {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "best_result": logger.best_result,
                "optimizer": optimizer.state_dict(),
                "args": options,
            },
            is_best,
            epoch,
            logger.output_directory,
        )


def _parse_args() -> argparse.Namespace:
    description = """
    This script launches a PENet training session with the specified set of
    options and dataset configuration. The intermediate results are logged
    in the specified log directory and can be visualized with tensorboard.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--session-name",
        type=str,
        required=True,
        help="Name of the sessions",
    )

    parser.add_argument(
        "-n",
        "--network-model",
        type=str,
        default="e",
        choices=["e", "pe"],
        help="Network model: 'e' = ENet, 'pe' = PENet",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="Number of total epochs to run (default: 100)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="Manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--start-epoch-bias",
        default=0,
        type=int,
        metavar="N",
        help="Manual epoch number bias (useful on restarts)",
    )
    parser.add_argument(
        "-c",
        "--criterion",
        metavar="LOSS",
        default="l2",
        choices=criteria.loss_names,
        help="Loss function.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-3,
        type=float,
        metavar="LR",
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-6,
        type=float,
        metavar="W",
        help="Weight decay",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="Print frequency",
    )
    parser.add_argument(
        "--resume",
        type=lambda p: Path(p).expanduser().resolve(),
        default=None,
        help="Path to the training checkpoint to resume from.",
    )
    parser.add_argument(
        "--data-folder",
        required=True,
        type=str,
        metavar="PATH",
        help="Full path to the directory containing the KITTI depth completion dataset",
    )
    parser.add_argument(
        "--data-folder-rgb",
        required=True,
        type=str,
        metavar="PATH",
        help="Full path to the directory containing the KITTI raw dataset.",
    )
    parser.add_argument(
        "--data-folder-save",
        required=True,
        type=str,
        metavar="PATH",
        help="Full path to the directory where the outputs should be saved.",
    )
    parser.add_argument(
        "--result",
        type=str,
        default=str(_DEFAULT_RESULTS_FOLDER),
        help="Full path to the results folder.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="rgbd",
        choices=input_options,
        help="Input type",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="select",
        choices=["select", "full"],
        help="full or select validation set",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.1,
        help="color jitter for images",
    )
    parser.add_argument(
        "--rank-metric",
        type=str,
        default="rmse",
        choices=[m for m in dir(Result()) if not m.startswith("_")],
        help="metrics for which best result is saved",
    )
    parser.add_argument(
        "-f",
        "--freeze-backbone",
        action="store_true",
        default=False,
        help="freeze parameters in backbone",
    )

    # random cropping
    parser.add_argument(
        "--not-random-crop",
        action="store_true",
        default=False,
        help="prohibit random cropping",
    )
    parser.add_argument(
        "-he",
        "--random-crop-height",
        default=320,
        type=int,
        metavar="N",
        help="random crop height",
    )
    parser.add_argument(
        "-w",
        "--random-crop-width",
        default=1216,
        type=int,
        metavar="N",
        help="random crop height",
    )

    # geometric encoding
    parser.add_argument(
        "-co",
        "--convolutional-layer-encoding",
        default="xyz",
        type=str,
        choices=["std", "z", "uv", "xyz"],
        help="information concatenated in encoder convolutional layers",
    )

    # dilated rate of DA-CSPN++
    parser.add_argument(
        "-d",
        "--dilation-rate",
        default=2,
        type=int,
        choices=[1, 2, 4],
        help="CSPN++ dilation rate",
    )

    options = parser.parse_args()
    options.use_rgb = "rgb" in options.input
    options.use_d = "d" in options.input
    options.use_g = "g" in options.input
    options.val_h = 352
    options.val_w = 1216
    return options


if __name__ == "__main__":
    options = _parse_args()
    train(options)
