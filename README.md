# PENet: Precise and Efficient Depth Completion
This repo is a fork of https://github.com/plr21/PENet_ICRA2021.
## Dependencies
An `environment.yml` file to create an Anaconda virtual environment is provided. Run:
```
conda env create -f environment.yml
```
inside the directory where you have cloned this repository.

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
