import glob
import os
import argparse
import skvideo.io
from tqdm import tqdm
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--frame_root',help='path to write the frames', type=str)
parser.add_argument('--video_path',help='path to the video folder', type=str)
parser.add_argument('--anno_path', help='path to the downloaded skeleton files', type=str)
args = parser.parse_args()

frame_root = args.frame_root
videos = sorted(glob.glob(os.path.join(args.video_path, '*.mp4')))
for video in videos:
    print(video)
    video_path = os.path.basename(video).replace('.mp4','')
    frame_path = os.path.join(frame_root,video_path)
    if not os.path.isdir(frame_path):
        os.mkdir(frame_path)

    anno_files = sorted(glob.glob(os.path.join(args.anno_path, video_path, '*.json')))
    image_num = len(anno_files)
    vidcap = skvideo.io.vread(video)
    assert image_num <= vidcap.shape[0]
    count=0
    for i in tqdm(range(image_num)):
        frame = vidcap[i]
        frame = Image.fromarray(frame)
        frame = frame.resize((256,256), Image.BICUBIC)
        name = os.path.join(frame_path,str(count).zfill(5)+'.png')
        frame.save(name)
        count+=1



