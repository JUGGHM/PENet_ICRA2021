import argparse
import typing
import logging
import torch

# Use GPU only if GPU is available and memory is more than 8GB
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
from pathlib import Path

from penet import vis_utils
from penet.s2d_predictor import PENetSparseToDensePredictor, get_model

LOGGER = logging.getLogger(__file__)


def _validate_file_path(file_path: Path) -> None:
    assert file_path.is_file(), f"{file_path} is not a valid file."


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

    penet_model = get_model(
        network_model, dilation_rate, conv_layer_encoding, checkpoint_path
    )
    s2d_predictor = PENetSparseToDensePredictor(penet_model)

    rgb = vis_utils.rgb_read(str(color_image_path))
    sparse_depth = vis_utils.depth_read(str(sparse_depth_image_path))

    dense_depth_image = s2d_predictor.predict(rgb, sparse_depth)

    # Save the depth image to disk
    vis_utils.save_depth_as_uint8colored(dense_depth_image, dense_depth_image_path)


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
