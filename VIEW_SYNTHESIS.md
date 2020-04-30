## ShapeNet Novel View Synthesis

### Introduction

We discuss the details about the novel view synthesis here. View synthesis requires generating novel views of objects or scenes based on arbitrary input views. 

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

### Dataset Preparation

We use the chair and car categories of the [ShapeNet](http://www.shapenet.org) dataset. Please download the preprocessed HDF5 files provided by [Multi-view to Novel view](https://github.com/shaohua0116/Multiview2Novelview): 

* [Car](https://drive.google.com/file/d/1vrZURHH5irKrxPFuw6e9mZ3wh2RqzFC9/view) (150G)

* [Chair](https://drive.google.com/file/d/1-IbmdJqi37JozGuDJ42IzOFG_ZNAksni/view) (14G)

And Put the files into `./dataset/ShapeNet`. 

We split the dataset using the same method as that of [Multi-view to Novel-View]((https://github.com/shaohua0116/Multiview2Novelview)). Pleast download the txt files from [here](https://drive.google.com/open?id=1v2mUFpAHklXQ0xnB_py7v02qq1P4yB03) and put these files into `./dataset/ShapeNet`.

### Training and Testing

In order to test the model, you can download the **trained weights** from:

* [ShapeNet Car](https://drive.google.com/open?id=1tD-s0gYPuXnvFY3X1l9ccsMLbhPEKm-b)
* [ShapeNet Chair](https://drive.google.com/open?id=1hiDxZQ6frYhfqtuvy9tXYjCfofVarcCs)

Put them into `./result/shape_net_car_checkpoints`,  `./result/shape_net_chair_checkpoints` respectively.

Then you can run the following example code to obtain the generated results.

```bash
python test.py \
--name=shape_net_car_checkpoints \
--model=shapenet \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=shapenet \
--sub_dataset=car \
--dataroot=./dataset/ShapeNet \
--results_dir=./eval_results/shape_net_car \
--checkpoints_dir=result 
```



If you want to train our model, you can run the following example code

```bash
# First, pre-train the Flow Field Estimator.
python train.py \
--name=shape_net_car \
--model=shapenetflow \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=shapenet \
--sub_dataset=car \
--dataroot=./dataset/ShapeNet

# Then, train the whole model in an end-to-end manner.
python train.py \
--name=shape_net_car \
--model=shapenet \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=shapenet \
--sub_dataset=car \
--dataroot=./dataset/ShapeNet \
--continue_train
```

The visdom is required to show the temporary results. You can access these results with:

```html
http://localhost:8096
```

 

