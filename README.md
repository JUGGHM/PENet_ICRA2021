## Important: About Improper Inference Time Statistics
The model's inference time is not properly reported in the original paper. This is because the original code ignores CUDA's asynchronous execution on CPU and GPU. To measure the inference time more precisely, the processes should be synchronized before recording current time:
```
torch.cuda.synchronize()
```
We restimated the inference time of following open-source models:

|methods|runtime_not_synchronized|runtime_synchronized|
|:----:|:----:|:----:|
|PENet|0.032s|0.161s|
|ENet|0.019s|0.064s|
|[NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20)|0.127s|0.130s|
|[ACMNet](https://github.com/sshan-zhao/ACMNet)|0.330s|0.350s|
|[DeepLiDAR](https://github.com/JiaxiongQ/DeepLiDAR)|0.051s|0.351s|
|[MSG-CHN](https://github.com/anglixjtu/msg_chn_wacv20)|0.011s|0.035s|
|[FusionNet](https://github.com/wvangansbeke/Sparse-Depth-Completion)|0.022s|0.029s|

We thank [wdjose](https://github.com/JUGGHM/PENet_ICRA2021/issues/4) for pointing out this problem. In addition,  ENet is more recommanded for real-time applications.

# PENet: Precise and Efficient Depth Completion
This repo is the PyTorch implementation of our paper to appear in ICRA2021 on ["Towards Precise and Efficient Image Guided Depth Completion"](https://arxiv.org/abs/2103.00783), developed by
Mu Hu, Shuling Wang, Bin Li, Shiyu Ning, Li Fan, and [Xiaojin Gong](https://person.zju.edu.cn/en/gongxj) at Zhejiang University and Huawei Shanghai.

Create a new issue for any code-related questions. Feel free to direct me as well at muhu@zju.edu.cn for any paper-related questions.

## Results
+ The proposed full model ranks 1st in the [KITTI depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) online leaderboard at the time of submission.
+ It infers much faster than most of the top ranked methods.
<div align=center><img src="https://github.com/JUGGHM/PENet_ICRA2021/blob/main/images/Comparison.png" width = "100%" height = "100%" /></div>

+ Both ENet and PENet can be trained thoroughly on 2x11G GPU.
+ Our network is trained with the KITTI dataset alone, not pretrained on Cityscapes or other similar driving dataset (either synthetic or real).

## Method
### A Strong Two-branch Backbone
#### Revisiting the popular two-branch architecture
<div align=center><img src="https://github.com/JUGGHM/PENet_ICRA2021/blob/main/images/Backbone.png" width = "100%" height = "100%" /></div>

The two-branch backbone is designed to thoroughly exploit color-dominant and depth-dominant information from
their respective branches and make the fusion of two modalities effective. Note that it is the depth prediction result
obtained from the color-dominant branch that is input to the depth-dominant branch, not a guidance map like those in [DeepLiDAR](https://github.com/JiaxiongQ/DeepLiDAR) and [FusionNet](https://github.com/wvangansbeke/Sparse-Depth-Completion).

#### Geometric convolutional Layer
<div align=center><img src="https://github.com/JUGGHM/PENet_ICRA2021/blob/main/images/Geometric_Encoding.png" width = "60%" height = "60%" /></div>

To encode 3D geometric information, it simply augments a conventional convolutional layer via concatenating a 3D position map to the layer’s input.

### Dilated and Accelerated CSPN++
#### Dilated CSPN
<div align=center><img src="https://github.com/JUGGHM/PENet_ICRA2021/blob/main/images/Dilated_CSPN.png" width = "60%" height = "60%" /></div>

We introduce a dilation strategy similar to the well known dilated convolutions to enlarge the propagation neighborhoods.

#### Accelerated CSPN
<div align=center><img src="https://github.com/JUGGHM/PENet_ICRA2021/blob/main/images/Accelerated_CSPN.png" width = "100%" height = "100%" /></div>

We design an implementation that makes the propagation from each neighbor truly parallel, which greatly accelerates the propagation procedure.

## Contents
1. [Dependency](#dependency)
0. [Data](#data)
0. [Trained Models](#trained-models)
0. [Commands](#commands)
0. [Citation](#citation)


## Dependency
Our released implementation is tested on.
+ Ubuntu 16.04
+ Python 3.7.4 (Anaconda 2019.10)
+ PyTorch 1.3.1 / torchvision 0.4.2
+ NVIDIA CUDA 10.0.130
+ 4x NVIDIA GTX 2080 Ti GPUs

```bash
pip install numpy matplotlib Pillow
pip install scikit-image
pip install opencv-contrib-python==3.4.2.17
```

## Data
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset and [KITTI Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php) Dataset from their websites.
The overall data directory is structured as follows:
```
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
```

```
├── kitti_raw
|   ├── 2011_09_26
|   ├── 2011_09_28
|   ├── 2011_09_29
|   ├── 2011_09_30
|   ├── 2011_10_03
```

## Trained Models
Download our pre-trained models:
- PENet (*i.e.*, the proposed full model with dilation_rate=2): [Download Here](https://drive.google.com/file/d/1RDdKlKJcas-G5OA49x8OoqcUDiYYZgeM/view?usp=sharing)
- ENet (*i.e.*, the backbone): [Download Here](https://drive.google.com/file/d/1TRVmduAnrqDagEGKqbpYcKCT307HVQp1/view?usp=sharing)

 Note that we don't need to decompress the pre-trained models. Just load the files of .pth.tar format directly.

## Commands
A complete list of training options is available with
```bash
python main.py -h
```
### Training
![Training Pipeline](https://github.com/JUGGHM/PENet_ICRA2021/blob/main/images/Training.png "Training")

Here we adopt a multi-stage training strategy to train the backbone, DA-CSPN++, and the full model progressively. However, end-to-end training is feasible as well.

1. Train ENet (Part Ⅰ)
```bash
CUDA_VISIBLE_DEVICES="0,1" python main.py -b 6 -n e
# -b for batch size
# -n for network model
```

2. Train DA-CSPN++ (Part Ⅱ)
```bash

CUDA_VISIBLE_DEVICES="0,1" python main.py -b 6 -f -n pe --resume [enet-checkpoint-path]
# -f for freezing the parameters in the backbone
# --resume for initializing the parameters from the checkpoint
```

3. Train PENet (Part Ⅲ)
```bash
CUDA_VISIBLE_DEVICES="0,1" python main.py -b 10 -n pe -he 160 -w 576 --resume [penet-checkpoint-path]
# -he, -w for the image size after random cropping
```

### Evalution
```bash
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 -n e --evaluate [enet-checkpoint-path]
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 -n pe --evaluate [penet-checkpoint-path]
# test the trained model on the val_selection_cropped data
```

### Test
```bash
CUDA_VISIBLE_DEVICES="0" python main.py -b 1 -n pe --evaluate [penet-checkpoint-path] --test
# generate and save results of the trained model on the test_depth_completion_anonymous data
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
