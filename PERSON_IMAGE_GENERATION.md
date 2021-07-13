## Pose-Guided Person Image Generation

The **Training and Testing details** of the pose-guided person image generation task are provided here.

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/80702562-dd337d00-8b13-11ea-95a1-1f86610a2335.png' width='700'/>
</p>
<p align='center'> 
</p>

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

- ~~We calculate the LPIPS scores using the code provided by the official repository [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity). Please clone their repository and put the folder **PerceptualSimilarity** to the folder [script](./script).~~
- This script is deprecated as the official repository of [LPIPS](https://github.com/richzhang/PerceptualSimilarity) has been updated. Please refer to their git for the evaluation.
- For FID, the real data distributions are calculated over the whole training set. 



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

 
