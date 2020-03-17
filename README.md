<p align='center'>
  <b>
    <a href="https://renyurui.github.io/GFLA-web"> Website</a>
    | 
    <a href="https://arxiv.org/abs/2003.00696">ArXiv</a>
    | 
    <a href="./README.md/#other-application">Application</a>
  </b>
</p> 

# Global-Flow-Local-Attention

The source code for our paper "[Deep Image Spatial Transformation for Person Image Generation](https://arxiv.org/abs/2003.00696)" (to appear in CVPR2020)

Our model can be ﬂexibly applied to tasks requiring spatial transformations such as:

* **Pose-Guided Person Image Generation:**

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/75610977-29346680-5b51-11ea-900e-c24eee54ddfb.png' width='700'/>
</p>
<p align='center'> 
  <b>Left:</b> generated results of our model; <b>Right:</b> Input source images.
</p>



* **Pose-Guided Person Image Animation**

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/75778006-fe852080-5d91-11ea-8e76-dd87f6c021f7.gif' width='900'/>
</p>
<p align='center'> 
  <b>From Left to Right</b>: Real Video; Extracted Pose; Animation Results.
</p>


* **Face Image Animation**

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/75610997-410bea80-5b51-11ea-9492-5d7b911bf450.gif' width='300'/>
  <img src='https://user-images.githubusercontent.com/30292465/75611009-5ed94f80-5b51-11ea-83a8-a30827f9f52e.gif' width='300'/>
</p>
<p align='center'> 
  <b>Left</b>: Input image; <b>Right</b>: Output results.
</p>



* **View Synthesis**

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/75611552-88e14080-5b56-11ea-9a1c-d29bdd7f7ef1.png' width='100'/>
  <img src='https://user-images.githubusercontent.com/30292465/75611012-71ec1f80-5b51-11ea-89f1-804a57d0112a.gif' width='300'/>
    <img src='https://user-images.githubusercontent.com/30292465/75611599-d362bd00-5b56-11ea-9570-b47fdd0775cb.png' width='100'/>
  <img src='https://user-images.githubusercontent.com/30292465/75611018-83352c00-5b51-11ea-9f32-3e3ed7bf9f32.gif' width='300'/>
</p>
<p align='center'> 
  <b>Form Left to Right:</b> 
  Input image, 
  Results of <a href="https://arxiv.org/abs/1605.03557">Appearance Flow</a>, 
  Results of Ours, Ground-truth images.
</p>

## News
* 2020.3.15 We upload the code and trained models of the [Face Animation](./IMAGE_ANIMATION.md) and [View Synthesis](VIEW_SYNTHESIS.md)!
* 2020.3.3   [Project Website](https://renyurui.github.io/GFLA-web) and [Paper](https://arxiv.org/abs/2003.00696) are avaliable!
* 2020.2.29  Code for PyTorch is available now!



## Installation

#### Requirements

* Python 3
* pytorch (1.0.0)
* CUDA
* visdom

#### Conda installation

```bash
# 1. Create a conda virtual environment.
conda create -n gfla python=3.6 -y
source activate gfla

# 2. Install dependency
pip install -r requirement.txt

# 3. Build pytorch Custom CUDA Extensions
./setup.sh
```

**Note**: The current code is tested with Tesla V100. **If you use a different GPU, you may need to select correct `nvcc_args` for your GPU when you buil Custom CUDA Extensions**. Comment or Uncomment `--gencode` in [block_extractor/setup.py](./model/networks/block_extractor/setup.py), [local_attn_reshape/setup.py](./model/networks/local_attn_reshape/setup.py), and  [resample2d_package/setup.py](./model/networks/resample2d_package/setup.py). Please check [here](https://medium.com/@patrickorcl/compile-with-nvcc-3566fbdfdbf) for details.



## Pre-Trained Models

We provide the pre-trained weights of our models. These resources can be downloaded from:

* **Pose-Guided Person Image Generation** 
  * **[Fashion](https://drive.google.com/open?id=1r1di3JFgaxqbyGzuRKDhNsrROiNL0s3r)**
  * **[Market](https://drive.google.com/open?id=1be_PY61HrVLg2CUOvtNEqmaLtmY83Muw)**
* **Pose Guided Person Image Animation**
  * Coming Soon 
* **Face Image Animation**
  * **[Face Animation](https://drive.google.com/open?id=1DwOR4F7dxEfqZTdQUvONPbJYJd3auTzF)**
* **Novel View Synthesis**
  * **[ShapeNet Car](https://drive.google.com/open?id=1tD-s0gYPuXnvFY3X1l9ccsMLbhPEKm-b)**
  * **[ShapeNet Chair](https://drive.google.com/open?id=1hiDxZQ6frYhfqtuvy9tXYjCfofVarcCs)**



## Pose-Guided Person Image Generation

### Dataset

#### Market1501

- Download the Market-1501 dataset from [here](http://www.liangzheng.com.cn/Project/project_reid.html). Rename **bounding_box_train** and **bounding_box_test** as **train** and **test**, and put them under the `./dataset/market` directory
- Download train/test key points annotations from [Google Drive](https://drive.google.com/open?id=1UAOyP-ZAKpMUoUbtXFST1AlmvQuV1uff) including **market-pairs-train.csv**, **market-pairs-test.csv**, **market-annotation-train.csv**, **market-annotation-train.csv**. Put these files under the  `./dataset/market` directory.

#### DeepFashion

- Download `img_highres.zip` of the DeepFashion Dataset from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Then put the obtained folder **img_highres** under the `./dataset/fashion` directory. 

- Download train/test key points annotations and the dataset list from [Google Drive](https://drive.google.com/open?id=1BX3Bxh8KG01yKWViRY0WTyDWbJHju-SL) including **fashion-pairs-train.csv**, **fashion-pairs-test.csv**, **fashion-annotation-train.csv**, **fashion-annotation-train.csv,** **train.lst**, **test.lst**. Put these files under the  `./dataset/fashion` directory.

- Run the following code to split the train/test dataset.

  ```bash
  python script/generate_fashion_datasets.py
  ```

  

### Evaluation

**Download the trained weights from [Fashion](https://drive.google.com/open?id=1r1di3JFgaxqbyGzuRKDhNsrROiNL0s3r),  [Market](https://drive.google.com/open?id=1be_PY61HrVLg2CUOvtNEqmaLtmY83Muw)**. Put the obtained checkpoints under `./result/pose_fashion_checkpoints` and `./result/pose_market_checkpoints` respectively.

Run the following codes to obtain the pose-based transformation results.

```bash
# Test the DeepFashion dataset 
python test.py \
--name=pose_fashion_checkpoints \
--model=pose \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=fashion \
--dataroot=./dataset/fashion \
--results_dir=./eval_results/fashion

# Test the market dataset
python test.py \
--name=pose_market_checkpoints \
--model=pose \
--attn_layer=2 \
--kernel_size=2=3 \
--gpu_id=0 \
--dataset_mode=market \
--dataroot=./dataset/market \
--results_dir=./eval_results/market
```

You can use the provided evaluation codes to evaluate the performance of our models.

```bash
# evaluate the performance (FID and LPIPS scores) over the DeepFashion dataset.
CUDA_VISIBLE_DEVICES=0 python -m  script.metrics \
--gt_path=./dataset/fashion/test_256 \
--distorated_path=./eval_results/fashion \
--fid_real_path=./dataset/fashion/train_256 \
--name=./fashion

# evaluate the performance (FID and LPIPS scores) over the Market dataset.
CUDA_VISIBLE_DEVICES=0 python -m  script.metrics \
--gt_path=./dataset/market/test_12864 \
--distorated_path=./eval_results/market \
--fid_real_path=./dataset/market/train_12864 \
--name=./market_12864
```

**Note**: 

* We calculate the LPIPS scores using the code provided by the official repository [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity). Please clone their repository and put the folder **PerceptualSimilarity** to the folder [script](./script).
* For FID, the real data distributions are calculated over the whole training set. 



### Training

We train our model in stages. The **Flow Field Estimator** is ﬁrst trained to generate ﬂow ﬁelds. Then we train **the whole model** in an end-to-end manner. 

For example, If you want to train our model with the DeepFashion dataset. You can use the following code.

```bash
# First train the Flow Field Estimator.
python train.py \
--name=fashion \
--model=poseflownet \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=fashion \
--dataroot=./dataset/fashion 

# Then, train the whole model in an end-to-end manner.
python train.py \
--name=fashion \
--model=pose \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=fashion \
--dataroot=./dataset/fashion \
--continue_train
```

The visdom is required to show the temporary results. You can access these results with:

```html
http://localhost:8096
```

 

## Other Applications

Our model can be ﬂexibly applied to tasks requiring spatial transformation. We show two examples: **Image Animation** and **View Synthesis**.

### Image Animation

Given **an input source image** and **a guidance video sequence** depicting the structure movements, our model generating a video containing the speciﬁc movements. We use the real video of the [FaceForensics dataset](https://github.com/ondyari/FaceForensics). See [IMAGE_ANIMATION.md](./IMAGE_ANIMATION.md) for more details.



### View synthesis

View synthesis requires generating novel views of objects or scenes based on arbitrary input views. In this task, we use the car and chair categories of the [ShapeNet dataset](https://www.shapenet.org). See [VIEW_SYNTHESIS.md](VIEW_SYNTHESIS.md) for more details.



## Citation

```tex
@article{ren2020deep,
  title={Deep Image Spatial Transformation for Person Image Generation},
  author={Ren, Yurui and Yu, Xiaoming and Chen, Junming and Li, Thomas H and Li, Ge},
  journal={arXiv preprint arXiv:2003.00696},
  year={2020}
}
```

## Acknowledgement 

We build our project base on [Vid2Vid](https://github.com/NVIDIA/vid2vid). Some dataset preprocessing methods are derived from [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer).