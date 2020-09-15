<p align='center'>
  <b>
    <a href="https://renyurui.github.io/GFLA-web"> Website</a>
    | 
    <a href="https://arxiv.org/abs/2003.00696">ArXiv</a>
    | 
    <a href="#Get-Start">Get Start</a>
  </b>
</p> 


# Global-Flow-Local-Attention

The source code for our paper "[Deep Image Spatial Transformation for Person Image Generation](https://arxiv.org/abs/2003.00696)" (CVPR2020)

We propose a Global-Flow Local-Attention Model for deep image spatial transformation. Our model can be ﬂexibly applied to tasks such as:

* **Pose-Guided Person Image Generation:**

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/75610977-29346680-5b51-11ea-900e-c24eee54ddfb.png' width='700'/>
</p>
<p align='center'> 
  <b>Left:</b> generated results of our model; <b>Right:</b> Input source images.
</p>



* **Pose-Guided Person Image Animation**

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/80788758-9352a180-8bbc-11ea-958d-d604e5cea1b9.gif' width='800'/>
</p>
<p align='center'> 
  <b>Left most</b>: Skeleton Squences. <b>The others</b>: Animation Results.
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
* 2020.4.30 Several demos are provided for quick exploration.

* 2020.4.29 Code for Pose-Guided Person Image Animation is avaliable now!

* 2020.3.15 We upload the code and trained models of the [Face Animation](./IMAGE_ANIMATION.md) and [View Synthesis](VIEW_SYNTHESIS.md)!

* 2020.3.3   [Project Website](https://renyurui.github.io/GFLA-web) and [Paper](https://arxiv.org/abs/2003.00696) are avaliable!

* 2020.2.29  Code for PyTorch is available now!

  

## Colab Demo

For a quick exploration of our model, find the online [colab demo](https://colab.research.google.com/drive/1kDMnB9IsnuWa_KFddOXC21OD8m_Hkpow).



## Get Start

### 1) Installation

**Requirements**

* Python 3
* pytorch (1.0.0)
* CUDA
* visdom

**Conda installation**

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



### 2) Download Resources

We provide **the pre-trained weights**  of our model. The resources are listed as following:

* **Pose-Guided Person Image Generation** 

  Google Drive: **[Fashion](https://drive.google.com/open?id=1r1di3JFgaxqbyGzuRKDhNsrROiNL0s3r)** | **[Market](https://drive.google.com/open?id=1be_PY61HrVLg2CUOvtNEqmaLtmY83Muw)**

  OneDrive: **[Fashion](https://1drv.ms/f/s!ArirMHnmz_frlAOL5s8lQ6eCfsNb)** | **[Market](https://1drv.ms/f/s!ArirMHnmz_frlAaL5s8lQ6eCfsNb)**

* **Pose Guided Person Image Animation**

  Google Drive: **[FashionVideo](https://drive.google.com/drive/folders/14bdd02GuR1dSTGAUkO_n4Xn0RjJXmdXV?usp=sharing)** | **[iPER](https://drive.google.com/drive/folders/11660gR9qgAdJrcnfjiaBxcb8cXDWx2uf?usp=sharing)**

  OneDrive: **[FashionVideo](https://1drv.ms/f/s!ArirMHnmz_frk3SL5s8lQ6eCfsNb)** | **[iPER](https://1drv.ms/f/s!ArirMHnmz_frk3iL5s8lQ6eCfsNb)**

* **Face Image Animation**

  Google Drive: **[Face Animation](https://drive.google.com/open?id=1DwOR4F7dxEfqZTdQUvONPbJYJd3auTzF)**

  OneDrive: **[Face_Animation](https://1drv.ms/f/s!ArirMHnmz_frk3-L5s8lQ6eCfsNb)**

* **Novel View Synthesis**

  Google Drive: **[ShapeNet Car](https://drive.google.com/open?id=1tD-s0gYPuXnvFY3X1l9ccsMLbhPEKm-b)** | **[ShapeNet Chair](https://drive.google.com/open?id=1hiDxZQ6frYhfqtuvy9tXYjCfofVarcCs)**

  OneDrive: **[ShapeNet_Car](https://1drv.ms/f/s!ArirMHnmz_frlAqL5s8lQ6eCfsNb)** | **[ShapeNet_Chair](https://1drv.ms/f/s!ArirMHnmz_frlAyL5s8lQ6eCfsNb)**

Download the **Per-Trained Models**  and the **Demo Images** by running the following code:

``` bash
./download.sh
```

### 3) Pose-Guided Person Image Generation

The Pose-Guided Person Image Generation task is to transfer a source person image to a target pose. 

Run the demo of this task:

``` bash
python demo.py \
--name=pose_fashion_checkpoints \
--model=pose \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=fashion \
--dataroot=./dataset/fashion \
--results_dir=./demo_results/fashion
```

For more training and testing details, please find the [PERSON_IMAGE_GENERATION.md](./PERSON_IMAGE_GENERATION.md)

### 4) Pose-Guided Person Image Animation

The Pose-Guided Person Image Animation task generates a video clip from a still source image according to a driving target sequence.  We further model the temporal consistency for this task.

Run the the demo of this task:

``` bash
python demo.py \
--name=dance_fashion_checkpoints \
--model=dance \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=dance \
--sub_dataset=fashion \
--dataroot=./dataset/danceFashion \
--results_dir=./demo_results/dance_fashion \
--test_list=val_list.csv
```

For more training and testing details, please find the [PERSON_IMAGE_ANIMATION.md.](./PERSON_IMAGE_ANIMATION.md)

### 5) Face Image Animation

Given an input source image and a guidance video sequence depicting the structure movements, our model generating a video containing the speciﬁc movements. 

Run the the demo of this task:

``` bash
python demo.py \
--name=face_checkpoints \
--model=face \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=face \
--dataroot=./dataset/FaceForensics \
--results_dir=./demo_results/face 
```

We use the real video of the [FaceForensics dataset](https://github.com/ondyari/FaceForensics). See [FACE_IMAGE_ANIMATION.md](./FACE_IMAGE_ANIMATION.md) for more details.

### 6) Novel View Synthesis

View synthesis requires generating novel views of objects or scenes based on arbitrary input views. 

In this task, we use the car and chair categories of the [ShapeNet dataset](https://www.shapenet.org). See [VIEW_SYNTHESIS.md](VIEW_SYNTHESIS.md) for more details.

## Citation

```tex
@article{ren2020deep,
  title={Deep Image Spatial Transformation for Person Image Generation},
  author={Ren, Yurui and Yu, Xiaoming and Chen, Junming and Li, Thomas H and Li, Ge},
  journal={arXiv preprint arXiv:2003.00696},
  year={2020}
}

@article{ren2020deep,
  title={Deep Spatial Transformation for Pose-Guided Person Image Generation and Animation},
  author={Ren, Yurui and Li, Ge and Liu, Shan and Li, Thomas H},
  journal={IEEE Transactions on Image Processing},
  year={2020},
  publisher={IEEE}
}
```

## Acknowledgement 

We build our project base on [Vid2Vid](https://github.com/NVIDIA/vid2vid). Some dataset preprocessing methods are derived from [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer).
