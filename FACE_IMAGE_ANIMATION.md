## Face Image Animation

### Introduction

We discuss the details about the Face Image Animation here. We use the edge maps as the structure guidance of target frames. Given **an input reference image** and **a guidance video sequence**, our model generating a video containing the speciﬁc movements. 

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/30292465/75610997-410bea80-5b51-11ea-9492-5d7b911bf450.gif' width='300'/>
  <img src='https://user-images.githubusercontent.com/30292465/75611009-5ed94f80-5b51-11ea-83a8-a30827f9f52e.gif' width='300'/>
</p>
<p align='center'> 
  <b>Left</b>: Input image; <b>Right</b>: Output results.
</p>

### Dataset Preparation

We use the real video of the [FaceForensics dataset](https://github.com/ondyari/FaceForensics). It contains 1000 Videos in total. We use 900 videos for training and 100 videos for testing. 

* Crop the videos so that the movement of faces is dominant in the videos. 
* Extract video frames and put the images into the forder `dataset/FaceForensics/train_data`
* Download face key point extractor `shape_predictor_68_face_landmarks.dat` form [here](https://github.com/davisking/dlib-models). Put the file under the forder `dataset/FaceForensics`. 
* Run the following code to generate the keypoint files for the extracted frames.

```bash
python script/obtain_face_kp.py
```

#### Training and Testing

Run the Following code to train our model.

```bash
python train.py \
--name=face \
--model=face \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0,1 \
--dataset_mode=face \
--dataroot=./dataset/FaceForensics\
```

**You can download our trained model from [here](https://drive.google.com/open?id=1DwOR4F7dxEfqZTdQUvONPbJYJd3auTzF).** 

Put the obtained weights into `./result/face_checkpoints`. Then run the following code to generate the animation videos.

```bash
python test.py \
--name=face_checkpoints \
--model=face \
--attn_layer=2,3 \
--kernel_size=2=5,3=3 \
--gpu_id=0 \
--dataset_mode=face \
--dataroot=./dataset/FaceForensics \
--results_dir=./eval_results/face \
--nThreads=1
```

The visdom is required to show the temporary results. You can access these results with:

```html
http://localhost:8096
```

 