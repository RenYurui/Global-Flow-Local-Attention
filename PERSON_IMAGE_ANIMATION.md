## Pose-Guided Person Image Animation

The **details** of the person image animation task are provided here.

The temporal consistency is modeled in this task. Specifically, the noisy poses extracted by the popular pose extraction methods are first prepossessed by a Motion Extraction Network to obtain clean poses. Then we generate the final animation results in a recurrent manner.  Technical Report is coming soon.

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/80794884-943fff00-8bcd-11ea-8287-91489b86deff.gif' width='800'/>
</p>
<p align='center'> 
  <b>From Left to Right</b>: Skeleton Squences. Propossessed Skeleton Seqences; Animation Results.
</p>

###Dataset

We provide the [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) extraction results of these datasets. Meanwhile, the prepossessed clean poses are also avaliable. Please use the following code to download these resources.

``` bash
./script/download_animation_skeletons.sh
```

#### FashionVideo

* Follow the instruction on dataset downloading [here](https://vision.cs.ubc.ca/datasets/fashion/). Extracted images from the original videos and resize them as 256 x 256. 
* Extract human poses using [Alphapose](https://github.com/MVIG-SJTU/AlphaPose). Jump this step if you have downloaded our provided resources.
* Follow the provided demo datasets `./dataset/danceFasion/val_256` (download using `./download.sh`) to build your training and testing set `./dataset/danceFasion/train_256`, `./dataset/danceFasion/test_256`.

#### iPER

* This dataset can be download from [here](https://svip-lab.github.io/project/impersonator). Extracted images from the original videos and resize them as 256 x 256. 
* Extract human poses using [Alphapose](https://github.com/MVIG-SJTU/AlphaPose).  Jump this step if you have downloaded our provided resources.
* Follow the provided demo datasets `./dataset/danceFasion/val_256` (download using `./download.sh`) to build your training and testing set `./dataset/iPER/train_256`, `./dataset/iPER/test_256`.



### Testing

**Download the trained weights from [FashionVideo](https://drive.google.com/drive/folders/14bdd02GuR1dSTGAUkO_n4Xn0RjJXmdXV?usp=sharing) and [iPER](https://drive.google.com/drive/folders/11660gR9qgAdJrcnfjiaBxcb8cXDWx2uf?usp=sharing)**. Put the obtained checkpoints under `./result/dance_fashion_checkpoints` and `./result/dance_iper_checkpoints` respectively.

Run the following codes to obtain the animation results.

``` bash
# test on the fashionVideo dataset 
python test.py \
--name=dance_fashion_checkpoints \
--model=dance \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=dance \
--sub_dataset=fashion \
--dataroot=./dataset/danceFashion \
--results_dir=./eval_results/dance_fashion \
--checkpoints_dir=result

# test on the iper dataset
python test.py \
--name=dance_iper_checkpoints \
--model=dance \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=dance \
--sub_dataset=iper \
--dataroot=./dataset/iPER \
--results_dir=./eval_results/dance_iper \
--checkpoints_dir=result
```



### Motion Extraction Net

We provide the details of the motion extraction net to support training and testing on custom datasets. You do not need this if you only want to test on the FashionVideo and iPER dataset.

This network is used to prepossess the noisy skeletons extracted by some pose extraction models. We train this model using the Human36M dataset. We download the training ground-truth label `data_2d_h36m_gt.npz` from [here](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). The corresponding input label `data_2d_h36m_detectron_pt_coco.npz` is download from [here](https://github.com/facebookresearch/VideoPose3D/issues/2#issuecomment-444687031).

Use the following code to train this model

``` bash 
python train.py \
--name=keypoint \
--model=keypoint \
--gpu_id=2 \
--dataset_mode=keypoint \
--continue_train
```

We also provide the [trained weights](https://drive.google.com/drive/folders/1Tc1MkSuFnGv9a_TcANQcChK6IS43oKQo). Assuming that you want to smooth the skeleton sequences of the iPER training set, you can use the following code

``` bash
python test.py \
--name=dance_keypoint_checkpoints \
--model=keypoint \
--gpu_id=2 \
--dataset_mode=keypointtest \
--continue_train \
--dataroot=./dataset/iPER \
--sub_dataset=iper \
--results_dir=./dataset/iPER/train_256/train_video2d \
--eval_set=train
```













