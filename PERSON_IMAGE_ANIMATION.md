## Pose-Guided Person Image Animation

The **details** of the person image animation task are provided here.

Person image animation is to generate a video clip using a source person image and target pose skeletons. Compare with the pose-guided person image generation task, this task requires to model the temporal consistency. Therefore, we modify the model in two ways: the noisy poses extracted by the popular pose extraction methods are first prepossessed by a Motion Extraction Network to obtain clean poses. Then we generate the final animation results in a recurrent manner. The technical details are provided in this [paper](https://arxiv.org/abs/2008.12606).

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/80794884-943fff00-8bcd-11ea-8287-91489b86deff.gif' width='800'/>
</p>
<p align='center'> 
  <b>From Left to Right</b>: Skeleton Squences. Propossessed Skeleton Seqences; Animation Results.
</p>


### Dataset

Two datasets are used in this task: [The FashionVideo dataset](https://vision.cs.ubc.ca/datasets/fashion/) and [the iPER dataset](https://svip-lab.github.io/project/impersonator). 

* Download the videos of the datasets.

* We provide the [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) extraction results of these datasets. Meanwhile, the prepossessed clean poses are also available. Please use the following code to download these resources.

  ``` bash
  ./script/download_animation_skeletons.sh
  ```

* Extract the image frames and resize them as 256 x 256 using the following code

  ``` bash
  python ./script/extract_video_frames.py \
  --frame_root=[path to write the video frames] \
  --video_path=[path to the mp4 files] \
  --anno_path=[path to the previously downloaded skeletons]
  ```

Note: you can also extract the skeleton on your own. Please use the [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) algorithm and output the results with openpose format.

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

### Training on your custom dataset 

If you want to train the model on your own dataset, you need to first extract the skeletons using the pose extraction algorithm Alphapose. Then extract the clean skeletons from the noisy data using the **motion extraction net**. 

#### Motion Extraction Net

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

We also provide trained [weights](https://drive.google.com/drive/folders/1Tc1MkSuFnGv9a_TcANQcChK6IS43oKQo). Assuming that you want to smooth the skeleton sequences of the iPER training set, you can use the following code

``` bash
python test.py \
--name=dance_keypoint_checkpoints \
--model=keypoint \
--gpu_id=2 \
--dataset_mode=keypointtest \
--dataroot=[root path of your dataset] \
--sub_dataset=iper \
--results_dir=[path to save the results] \
--eval_set=[train/test/val]
```



After obtain the clean skeletons. You can train our model on your dataset using the following code. (Note: you need to modify the `dance_dataset.py` to add your dataset as a sub_set)

``` bash
python train.py \
--name=[name_of_the_experiment] \
--model=dance \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0,1 \
--dataset_mode=dance \
--sub_dataset=[iper/fashion/your_dataset_name] \
--dataroot=[your_dataset_root] \
--continue_train
```











