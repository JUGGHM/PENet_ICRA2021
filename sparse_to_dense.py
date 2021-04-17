import argparse
import typing
import logging

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

# Use GPU only if GPU is available and memory is more than 8GB
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from collections import namedtuple
from pathlib import Path

import CoordConv
import vis_utils
from dataloaders import transforms
from dataloaders import kitti_loader
from model import ENet, PENet_C1, PENet_C2, PENet_C4

LOGGER = logging.getLogger(__file__)

ModelConfig = namedtuple(
    "ModelConfig", ["network_model", "dilation_rate", "convolutional_layer_encoding"]
)

# Height and width of the input images
INPUT_DIMS = (352, 1216)


def _validate_file_path(file_path: Path) -> None:
    assert file_path.is_file(), f"{file_path} is not a valid file."


def _get_model(network_model: str, dilation_rate: int, conv_layer_encoding: str):
    config = ModelConfig(network_model, dilation_rate, conv_layer_encoding)
    model = None
    penet_accelerated = False
    if network_model == "e":
        model = ENet(config).to(device)
    else:
        if dilation_rate == 1:
            model = PENet_C1(config).to(device)
            penet_accelerated = True
        elif dilation_rate == 2:
            model = PENet_C2(config).to(device)
            penet_accelerated = True
        elif dilation_rate == 4:
            model = PENet_C4(config).to(device)
            penet_accelerated = True

    if penet_accelerated == True:
        model.encoder3.requires_grad = False
        model.encoder5.requires_grad = False
        model.encoder7.requires_grad = False

    return model


def _prepare_input(
    color_image_path: Path, sparse_depth_image_path: Path
) -> typing.Dict:
    # Get the calibration matrix
    K = kitti_loader.load_calib()

    rgb = vis_utils.rgb_read(str(color_image_path))
    d = vis_utils.depth_read(str(sparse_depth_image_path))

    heigth, width = INPUT_DIMS
    position = CoordConv.AddCoordsNp(heigth, width).call()

    to_tensor = transforms.ToTensor()

    # Create input data dict
    data_dict = {
        "rgb": to_tensor(rgb).float(),
        "d": to_tensor(d).float(),
        "gt": None,
        "g": None,
        "position": to_tensor(position).float(),
        "K": to_tensor(K).float(),
    }

    return {
        key: torch.unsqueeze(value, 0)
        for key, value in data_dict.items()
        if value is not None
    }


def sparse_to_dense(
    checkpoint_path: Path,
    color_image_path: Path,
    sparse_depth_image_path: Path,
    dense_depth_image_path: typing.Optional[Path],
    network_model: str,
    dilation_rate: int,
    conv_layer_encoding: str,
) -> None:
    _validate_file_path(checkpoint_path)
    _validate_file_path(color_image_path)
    _validate_file_path(sparse_depth_image_path)

    if dense_depth_image_path is None:
        dense_depth_image_path = (
            sparse_depth_image_path.stem + "_dense" + sparse_depth_image_path.suffix
        )

    model = _get_model(network_model, dilation_rate, conv_layer_encoding)
    # Load weights from checkpoint file
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    # Get the data
    input_data_dict = _prepare_input(color_image_path, sparse_depth_image_path)

    # Predict the dense depth map
    model.eval()
    if network_model == "e":
        _, _, pred_dense_depth_image = model(input_data_dict)
    else:
        pred_dense_depth_image = model(input_data_dict)

    # Save the depth image to disk
    vis_utils.save_depth_as_uint8colored(pred_dense_depth_image, dense_depth_image_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""
        Generate a dense depth map from a color image and a sparse depth map.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-rgb",
        "--color-image",
        required=True,
        type=lambda p: Path(p).expanduser().resolve(),
        help="Color image file path.",
    )

    parser.add_argument(
        "-pc",
        "--sparse-depth-image",
        required=True,
        type=lambda p: Path(p).expanduser().resolve(),
        help="Sparse depth map file path.",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        type=lambda p: Path(p).expanduser().resolve(),
        help="Path to the pytorch checkpoint file.",
    )

    parser.add_argument(
        "-o",
        "--dense-depth-image",
        type=lambda p: Path(p).expanduser().resolve(),
        default=None,
        help="Output dense depth map file path.",
    )

    parser.add_argument(
        "-n",
        "--network_model",
        type=str,
        choices=["e", "pe"],
        default=["pe"],
        help="Choose between ENet and PENet",
    )

    parser.add_argument(
        "-dr",
        "--dilation-rate",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help="Dilation rate used for depth prediction",
    )

    parser.add_argument(
        "-co",
        "--conv-layer-encoding",
        default="xyz",
        type=str,
        choices=["std", "z", "uv", "xyz"],
        help="information concatenated in encoder convolutional layers",
    )

    parser.add_argument(
        "-v",
        "--verbosity-level",
        type=int,
        choices=[logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
        default=logging.INFO,
        help="""Verbosity level:
        10 -> Debug
        20 -> Info
        30 -> Warning
        40 -> Error
        """,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(level=args.verbosity_level)

    sparse_to_dense(
        checkpoint_path=args.checkpoint,
        color_image_path=args.color_image,
        sparse_depth_image_path=args.sparse_depth_image,
        dense_depth_image_path=args.dense_depth_image,
        network_model=args.network_model,
        dilation_rate=args.dilation_rate,
        conv_layer_encoding=args.conv_layer_encoding,
    )
