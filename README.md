# Global-Flow-Local-Attention
The source code for paper "Deep Image Spatial Transformation for Person Image Generation" (to appear in CVPR2020)



## News

* 2020.2.29  Code for pytorch are available now!



## Installation

#### Requirements

* Python 3
* pytorch (1.0.0)
* CUDA
* visdom

#### Conda installation

```shell
# 1. Create a conda virtual environment.
conda create -n gfla python=3.6 -y
source activate gfla

# 2. Install dependency
pip install -r requirement.txt

# 3. Build pytorch Custom CUDA Extensions
./setup.sh
```

**Note**: The current code is tested with Tesla V100. If you use a different GPU, you may need to select correct `nvcc_args` for your GPU when you buil Custom CUDA Extensions. Comment or Uncomment `--gencode` in [block_extractor/setup.py](./model/networks/block_extractor/setup.py), [local_attn_reshape/setup.py](./model/networks/local_attn_reshape/setup.py), and  [resample2d_package/setup.py](./model/networks/resample2d_package/setup.py). Please check [here](https://medium.com/@patrickorcl/compile-with-nvcc-3566fbdfdbf) for details.



## Dataset

#### Market1501

- Download the Market-1501 dataset from [here](http://www.liangzheng.com.cn/Project/project_reid.html). Rename **bounding_box_train** and **bounding_box_test** as **train** and **test**, and put them under the [./dataset/market](./dataset/market) directory
- Download train/test key points annotations from [Google Drive](https://drive.google.com/open?id=1UAOyP-ZAKpMUoUbtXFST1AlmvQuV1uff) including **market-pairs-train.csv**, **market-pairs-test.csv**, **market-annotation-train.csv**, **market-annotation-train.csv**. Put these files under the  [./dataset/market](./dataset/market)  directory.

#### DeepFashion

- Download Fashion Dataset `img_highres.zip` from [In-shop Clothes Retrieval Benchmark](https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00). 

- Unzip `img_highres.zip`. Then put the obtained folder **img_highres** under [./dataset/fashion](./dataset/fashion) . You will need to ask for password from the [dataset maintainers](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html).

- Download train/test key points annotations and the dataset list from [] including **fashion-pairs-train.csv**, **fashion-pairs-test.csv**, **fashion-annotation-train.csv**, **fashion-annotation-train.csv,** **train.lst**, **test.lst**. Put these files under the  [./dataset/fashion](./dataset/fashion)  directory.

- run the following code to split the train/test dataset.

  ```python
  python script/generate_fashion_datasets.py
  ```

  

##Evaluation

**The trained weights can be downloaded from [Fashion](https://drive.google.com/open?id=1r1di3JFgaxqbyGzuRKDhNsrROiNL0s3r),  [Market](https://drive.google.com/open?id=1be_PY61HrVLg2CUOvtNEqmaLtmY83Muw),  Face (coming soon), ShapeNet(coming soon).**

Put the downloaded checkpoints to the floder [result](./result). Run the following codes to obtain the transformation results.

```python
# Test over the DeepFashion dataset 
python test.py \
--name=pose_fashion_checkpoints \
--model=pose \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=fashion \
--dataroot=./dataset/fashion \
--results_dir=./eval_results/fashion

# Test over the market dataset
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

You can use the provided evaluation code to evaluate the performance of our model.

```python
# evaluate the performace (FID and LPIPS scores) over the DeepFashion dataset.
CUDA_VISIBLE_DEVICES=0 python -m  script.metrics \
--gt_path=./dataset/fashion/test_256 \
--distorated_path=./eval_results/fashion \
--fid_real_path=./dataset/fashion/train_256 \
--name=./fashion

# evaluate the performace (FID and LPIPS scores) over the Market dataset.
CUDA_VISIBLE_DEVICES=0 python -m  script.metrics \
--gt_path=./dataset/market/test_12864 \
--distorated_path=./eval_results/market \
--fid_real_path=./dataset/market/train_12864 \
--name=./market_12864
```

**Note**: For FID, the real data distributions are calculated over the whole training set. 



##Training

We train our model in stages. The **Flow Field Estimator** is ﬁrst trained to generate ﬂow ﬁelds. Then we train **the whole model** in an end-to-end manner. 

For example, If you want to train our model using DeepFashion dataset. You can use the following code.

```python
# First train the Flow Field Estimator.
python train.py \
--name=fashion \
--model=pose_flow \
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

 

## Other Application

Our model can be used to solve other tasks requiring spatial trasformation such as **Image Animation** and **View Synthesis**.

Details will come soon.



