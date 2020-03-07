import glob
import os
import shutil
import numpy as np
from PIL import Image
import dlib
from imutils import face_utils
import warnings

folder='./dataset/FaceForensics/train_data/*'
folder_k='./dataset/FaceForensics/train_keypoint/*'
predictor_path = './dataset/FaceForensics/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


videos = sorted(glob.glob(folder))
for video in videos:
  txt_path = video.replace('train_data', 'train_keypoint')
  if not os.path.isdir(txt_path):
    os.mkdir(txt_path)
  images = sorted(glob.glob(os.path.join(video, '*.png')))
  for image in images:
    current = image.replace('train_data', 'train_keypoint').replace('.png','.txt')
    print(current)
    PIL_image=Image.open(image)
    PIL_image=np.asarray(PIL_image)

    dets = detector(PIL_image, 1)

    # loop over the face detections
    if len(dets)>1:
      warnings.warn("more than one faces detected in image: %s"%image)

    elif len(dets) == 0:
      warnings.warn("No faces detected in image: %s"%image)
      shutil.copy(previous, current)
      continue

    rect = dets[0]
    shape = predictor(PIL_image, rect)
    shape = face_utils.shape_to_np(shape).astype(np.int)
    np.savetxt(current, shape, fmt="%d,%d")
    previous = current
    

