import json
import numpy as np
import torch
from collections import namedtuple
from pathlib import Path
from typing import Dict, Union

from . import CoordConv
from . import models
from .dataloaders import transforms, kitti_loader

# Height and width of the input images
INPUT_DIMS = (352, 1216)

# Use GPU only if GPU is available and memory is more than 8GB
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Model params
ModelConfig = namedtuple(
    "ModelConfig", ["network_model", "dilation_rate", "convolutional_layer_encoding"]
)

def get_model(network_model: str, dilation_rate: int, conv_layer_encoding: str, checkpoint_file_path: Path):
    config = ModelConfig(network_model, dilation_rate, conv_layer_encoding)
    penet_accelerated = False
    if network_model == "e":
        model = models.ENet(config).to(device)
    else:
        if dilation_rate == 1:
            model = models.PENet_C1(config).to(device)
            penet_accelerated = True
        elif dilation_rate == 2:
            model = models.PENet_C2(config).to(device)
            penet_accelerated = True
        elif dilation_rate == 4:
            model = models.PENet_C4(config).to(device)
            penet_accelerated = True

    if penet_accelerated:
        model.encoder3.requires_grad = False
        model.encoder5.requires_grad = False
        model.encoder7.requires_grad = False
    
    # FIXME - This is nasty hack. The pretrained models available
    # on GitHub contain the whole model objects, not just the state dicts
    # (see https://github.com/pytorch/pytorch/issues/3678) so it expects
    # module metrics.py to be in the PYTHONPATH. So, since the source code
    # was moved into this module, we need this workaround to load the models.
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    checkpoint = torch.load(str(checkpoint_file_path), map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    return model

class PENetSparseToDensePredictor:

    def __init__(
        self, penet_model
    ):
        self.penet_model = penet_model
        self.to_tensor_functor = transforms.ToTensor()
        position = CoordConv.AddCoordsNp(INPUT_DIMS[0], INPUT_DIMS[1]).call()
        self.position_tensor = self.to_tensor_functor(position).float()
        # TODO: this is still hard-coded, make it a parameter
        camera_intrinsics = kitti_loader.load_calib()
        self.camera_intrinsics_tensor = self.to_tensor_functor(camera_intrinsics).float()

    @classmethod
    def from_config_file(cls, config_file_path: Path):
        with open(config_file_path, 'r') as j:
            config = json.load(j)

        checkpoint_file_path = Path(config["checkpoint"])
        assert checkpoint_file_path.is_file()
        network_model = config["network_model"]
        assert network_model in ["e", "pe"]
        dilation_rate = config["dilation_rate"]
        assert dilation_rate in [1,2,3,4]
        conv_layer_encoding = config["convolutional_layer_encoding"]
        assert conv_layer_encoding in ["xyz", "std"]

        # Get model data structure
        penet_model = get_model(network_model, dilation_rate, conv_layer_encoding, checkpoint_file_path)
        
        return cls(penet_model)


    def _prepare_input(
        self, rgb: np.ndarray, sparse_depth: np.ndarray,
    ) -> Dict[str, Union[None, np.ndarray]]:

        assert rgb.shape[:2] == INPUT_DIMS, f"rgb shape={rgb.shape[:2]} != {INPUT_DIMS} INPUT_DIMS"
        assert sparse_depth.shape[:2] == INPUT_DIMS

        if sparse_depth.ndim == 2:
            sparse_depth = np.expand_dims(sparse_depth, -1)

        # Create input data dict
        data_dict = {
            "rgb": self.to_tensor_functor(rgb).float(),
            "d": self.to_tensor_functor(sparse_depth).float(),
            "gt": None,
            "g": None,
            "position": self.position_tensor,
            "K": self.camera_intrinsics_tensor
        }

        return {
            key: torch.unsqueeze(value, 0).to(device)
            for key, value in data_dict.items()
            if value is not None
        }

    def predict(self, rgb: np.ndarray, sparse_depth: np.ndarray) -> np.ndarray:
        input_data_dict = self._prepare_input(rgb, sparse_depth)

        if isinstance(self.penet_model, models.ENet):
            _, _, pred_dense_depth_image = self.penet_model(input_data_dict)
        else:
            pred_dense_depth_image = self.penet_model(input_data_dict)

        return pred_dense_depth_image
        
