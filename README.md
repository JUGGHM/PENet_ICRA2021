# PEDC: Precise and Efficient Depth Completion
This repo is the PyTorch implementation of our paper to appear in ICRA2021 on ["Towards Precise and Efficient Image Guided Depth Completion"](https://arxiv.org/pdf/.pdf), developed by
Mu Hu, Shuling Wang, Bin Li, Shiyu Ning, Li Fan, and [Xiaojin Gong](https://person.zju.edu.cn/en/gongxj) at Zhejiang University and Huawei Shanghai.

## Precision and Efficiency
### Precision
The proposed full model ranks 1st in the [KITTI depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) online leaderboard at the time of submission.
### Inference Efficiency: Fast
It infers much faster than most of the top ranked methods.
### Training Efficiency: Small GPU Consuming and No Additional Dataset Required
Both ENet and PENet can be trained thoroughly on 2x11G GPUs.

Our network is trained with the KITTI dataset alone, without pretraining on Cityscapes or other similar driving dataset (either synthetic or real).

## A Strong Two-branch Backbone

## Dilated CSPN++

## Accelerated CSPN++



## Contents
1. [Dependency](#dependency)
0. [Data](#data)
0. [Trained Models](#trained-models)
0. [Commands](#commands)
0. [Citation](#citation)


## Dependency
This code was tested with Python 3 and PyTorch 1.0 on Ubuntu 16.04.
```bash
pip install numpy matplotlib Pillow
pip install torch torchvision # pytorch

# for self-supervised training requires opencv, along with the contrib modules
pip install opencv-contrib-python==3.4.2.16
```

## Data
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset from their website.
The overall data directory is structured as follows
```
|
├── kitti_depth
|   ├── depth
|   |   ├──data_depth_annotated
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_velodyne
|   |   |  ├── train
|   |   |  ├── val
|   |   ├── data_depth_selection
|   |   |  ├── test_depth_completion_anonymous
|   |   |  |── test_depth_prediction_anonymous
|   |   |  ├── val_selection_cropped
├── kitti_raw
|   ├── 2011_09_26
|   ├── 2011_09_28
|   ├── 2011_09_29
|   ├── 2011_09_30
|   ├── 2011_10_03
```

## Trained Models
Download our pre-trained models:
- PENet (*i.e.*, the proposed full model with dilation_rate=2): [Download Here](https://drive.google.com/file/d/1TRVmduAnrqDagEGKqbpYcKCT307HVQp1/view?usp=sharing)
- ENet (*i.e.*, the backbone): [Download Here](https://drive.google.com/file/d/1RDdKlKJcas-G5OA49x8OoqcUDiYYZgeM/view?usp=sharing)

## Commands
A complete list of training options is available with
```bash
python main.py -h
```
For instance,
```bash
# train with the KITTI semi-dense annotations, rgbd input, and batch size 1
python main.py --train-mode dense -b 1 --input rgbd

# train with the self-supervised framework, not using ground truth
python main.py --train-mode sparse+photo

# resume previous training
python main.py --resume [checkpoint-path]

# test the trained model on the val_selection_cropped data
python main.py --evaluate [checkpoint-path] --val select
```

## Citation
If you use our code or method in your work, please cite the following:

	@article{hu2020PENet,
		title={Towards Precise and Efficient Image Guided Depth Completion},
		author={Hu, Mu and Wang, Shuling and Li, Bin and Ning, Shiyu and Fan, Li and Gong, Xiaojin},
		booktitle={ICRA},
		year={2021}
	}

## Related Repositories
The original code framework is rendered from ["Self-supervised Sparse-to-Dense:  Self-supervised Depth Completion from LiDAR and Monocular Camera"](https://github.com/fangchangma/self-supervised-depth-completion). It is developed by [Fangchang Ma](http://www.mit.edu/~fcma/), Guilherme Venturelli Cavalheiro, and [Sertac Karaman](http://karaman.mit.edu/) at MIT.

The part of CoordConv is rendered from ["An intriguing failing of convolutional neural networks and the CoordConv"](https://github.com/mkocabas/CoordConv-pytorch).
