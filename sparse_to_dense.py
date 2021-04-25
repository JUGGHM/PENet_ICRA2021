import argparse
import typing
import logging

# Use GPU only if GPU is available and memory is more than 8GB
import torch

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import vis_utils
from pathlib import Path
from os import makedirs
from penet_s2d_predictor import PENetSparseToDensePredictor, get_model, INPUT_DIMS

LOGGER = logging.getLogger(__file__)

def _validate_file_path(file_path: Path) -> None:
    assert file_path.is_file(), f"{file_path} is not a valid file."

def _validate_folder_path(folder_path: Path) -> None:
    assert folder_path.is_dir(), f"{folder_path} is not a valid directory."

def _validate_img_depth_pair(color_img_path: Path, depth_img_path: Path) ->None:
    assert color_img_path.stem == depth_img_path.stem, (
        f"Inconsistent image {color_img_path} and depth {depth_img_path}")

def _image_depth_content_consistent(color_images: list, depth_images: list) -> bool:
    return len(color_images) == len(depth_images)

def _get_preprocess_crop_fcn(preprocess_crop_type: str):
    if preprocess_crop_type == "center":
        preprocess_crop = vis_utils.centercrop(INPUT_DIMS)
    else:
        preprocess_crop = vis_utils.bottomcrop(INPUT_DIMS)
    return preprocess_crop

def sparse_to_dense(
    checkpoint_path: Path,
    color_image_path: Path,
    sparse_depth_image_path: Path,
    dense_depth_image_path: typing.Optional[Path],
    network_model: str,
    dilation_rate: int,
    conv_layer_encoding: str,
    preprocess_crop_type: str
) -> None:
    _validate_file_path(checkpoint_path)
    _validate_file_path(color_image_path)
    _validate_file_path(sparse_depth_image_path)

    if dense_depth_image_path is None:
        dense_depth_image_path = (
            sparse_depth_image_path.stem + "_dense" + sparse_depth_image_path.suffix
        )

    # Preprocess transform
    preprocess_transform = _get_preprocess_crop_fcn(preprocess_crop_type)

    penet_model = get_model(
        network_model, dilation_rate, conv_layer_encoding, checkpoint_path
    )
    s2d_predictor = PENetSparseToDensePredictor(penet_model)

    rgb = preprocess_transform(vis_utils.rgb_read(str(color_image_path)))
    sparse_depth = preprocess_transform(vis_utils.depth_read(str(sparse_depth_image_path)))

    dense_depth_image = s2d_predictor.predict(rgb, sparse_depth)

    # Save the depth image to disk
    vis_utils.save_depth_as_uint8colored(dense_depth_image, dense_depth_image_path)

def sparse_to_dense_dataset(
    checkpoint_path: Path,
    color_images_dir: Path,
    sparse_depths_dir: Path,
    dense_depths_dir: typing.Optional[Path],
    network_model: str,
    dilation_rate: int,
    conv_layer_encoding: str,
    preprocess_crop_type: str,
) -> None:
    _validate_folder_path(color_images_dir)
    _validate_folder_path(sparse_depths_dir)

    if dense_depths_dir is None:
        dense_depths_dir = sparse_depths_dir.parent / "DenseDepthPEnet"
        if not dense_depths_dir.is_dir():
            makedirs(dense_depths_dir)
    else:
        _validate_folder_path(dense_depths_dir)

    # Preprocess transform
    preprocess_transform = _get_preprocess_crop_fcn(preprocess_crop_type)

    # Obtaining images and depths
    supported_img_formats = [".png"]
    supported_depth_formats = [".png", ".tiff"]
    for img_format in supported_img_formats:
        img_files = sorted(color_images_dir.glob("*"+img_format))
        if len(img_files) > 0:
            break
    for depth_format in supported_depth_formats:
        depth_files = sorted(sparse_depths_dir.glob("*"+depth_format))
        if len(depth_files) > 0:
            break
    assert len(img_files) == len(depth_files), (
        f"Number of color images and depth images is different. {len(img_files)}!={len(depth_files)}")

    # Loading PENet
    penet_model = get_model(
        network_model, dilation_rate, conv_layer_encoding, checkpoint_path
    )
    s2d_predictor = PENetSparseToDensePredictor(penet_model)

    for img_file, depth_file in zip(img_files, depth_files):
        _validate_img_depth_pair(img_file, depth_file)
        dense_depth_file_f = dense_depths_dir / (img_file.stem+".tiff")
        dense_depth_file_i = dense_depths_dir / (img_file.stem+".png")

        # Evaluate PENET
        rgb = preprocess_transform(vis_utils.rgb_read(str(img_file)))
        sparse_depth = preprocess_transform(vis_utils.depth_read(str(depth_file)))
        dense_depth_image = s2d_predictor.predict(rgb, sparse_depth)

        # Save the depth image to disk
        vis_utils.save_depth_as_uint8colored(dense_depth_image, str(dense_depth_file_i))
        vis_utils.save_depth_as_floattiff(dense_depth_image, str(dense_depth_file_f))

        print(f"Instance read: {img_file.stem}")
        print(f"\tImage: {img_file}")
        print(f"\tDepth: {depth_file}")
        print(f"Output:")
        print(f"\tDense_f: {dense_depth_file_f}")
        print(f"\tDense_rgb: {dense_depth_file_i}")

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
        help="Color image file path / Color images directory.",
    )

    parser.add_argument(
        "-pc",
        "--sparse-depth-image",
        required=True,
        type=lambda p: Path(p).expanduser().resolve(),
        help="Sparse depth map file path / Sparse depth images directory.",
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
        help="Output dense depth map file path / Ouput dense depth maps directory.",
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
        choices=[1, 2, 4],
        default=2,
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
        "-cr",
        "--preprocess-crop",
        default = "bottom",
        type=str,
        choices=["center", "bottom"],
        help="resizing typeto match network size input"
    )

    parser.add_argument(
        "-sd",
        "--sparse-dataset",
        dest='sparse_dataset',
        action='store_true',
        help = "Run on sparse-depth dataset",
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

    if args.sparse_dataset:
        sparse_to_dense_dataset(
            checkpoint_path = args.checkpoint,
            color_images_dir = args.color_image,
            sparse_depths_dir = args.sparse_depth_image,
            dense_depths_dir = args.dense_depth_image,
            network_model = args.network_model,
            dilation_rate = args.dilation_rate,
            conv_layer_encoding = args.conv_layer_encoding,
            preprocess_crop_type=args.preprocess_crop
        )
        print("Finished predictions on dataset")
    else:
        sparse_to_dense(
            checkpoint_path=args.checkpoint,
            color_image_path=args.color_image,
            sparse_depth_image_path=args.sparse_depth_image,
            dense_depth_image_path=args.dense_depth_image,
            network_model=args.network_model,
            dilation_rate=args.dilation_rate,
            conv_layer_encoding=args.conv_layer_encoding,
            preprocess_crop_type=args.preprocess_crop
        )
        print("Finished prediction on image")
