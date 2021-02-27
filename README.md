# PENet
This repo is the PyTorch implementation of our paper to appear in ICRA2021 on ["Towards Precise and Efficient Image Guided Depth Completion"](https://arxiv.org/pdf/.pdf), developed by
Mu Hu, Shuling Wang, Bin Li, Shiyu Ning, Li Fan, and [Xiaojin Gong](https://person.zju.edu.cn/en/gongxj) at Zhejiang University and Huawei Shanghai.

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
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset from their website. Use the following scripts to extract corresponding RGB images from the raw dataset.
```bash
./download/rgb_train_downloader.sh
./download/rgb_val_downloader.sh
```
The downloaded rgb files will be stored in the `../data/data_rgb` folder. The overall code, data, and results directory is structured as follows (updated on Oct 1, 2019)
```
.
├── self-supervised-depth-completion
├── data
|   ├── data_depth_annotated
|   |   ├── train
|   |   ├── val
|   ├── data_depth_velodyne
|   |   ├── train
|   |   ├── val
|   ├── depth_selection
|   |   ├── test_depth_completion_anonymous
|   |   ├── test_depth_prediction_anonymous
|   |   ├── val_selection_cropped
|   └── data_rgb
|   |   ├── train
|   |   ├── val
├── results
```

## Trained Models
Download our trained models at http://datasets.lids.mit.edu/self-supervised-depth-completion to a folder of your choice.
- supervised training (i.e., models trained with semi-dense lidar ground truth): http://datasets.lids.mit.edu/self-supervised-depth-completion/supervised/
- self-supervised (i.e., photometric loss + sparse depth loss + smoothness loss): http://datasets.lids.mit.edu/self-supervised-depth-completion/self-supervised/

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

  The original framework is rendered from ["Self-supervised Sparse-to-Dense:  Self-supervised Depth Completion from LiDAR and Monocular Camera"](https://github.com/fangchangma/self-supervised-depth-completion). It is developed by [Fangchang Ma](http://www.mit.edu/~fcma/), Guilherme Venturelli Cavalheiro, and [Sertac Karaman](http://karaman.mit.edu/) at MIT.

## Citation
If you use our code or method in your work, please cite the following:

	@article{hu2020PENet,
		title={Towards Precise and Efficient Image Guided Depth Completion},
		author={Hu, Mu and Wang, Shuling and Li, Bin and Ning, Shiyu and Fan, Li and Gong, Xiaojin},
		booktitle={ICRA},
		year={2021}
	}
