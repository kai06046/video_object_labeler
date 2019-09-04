import cv2
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='Video to image converter')
parser.add_argument('--video', help='path of video that to be converted')
args = parser.parse_args()

video_path = args.video
root = "/".join(video_path.split('/')[:-1])
video_name = video_path.split('/')[-1].split('.')[0]
video = cv2.VideoCapture(video_path)
img_root = os.path.join(root, video_name)
if not os.path.exists(img_root):
    os.makedirs(img_root)

image_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
image_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
half_image_w = int(image_w/2)
half_image_h = int(image_h/2)

print('Start converting...')

n_frame = 0
while True:
    ok, frame = video.read()
    if not ok:
        break
    if image_w > 1080:
        frame = cv2.resize(frame, (half_image_w, half_image_h))
    name = '%s/%06d.jpg' % (img_root, n_frame)
    cv2.imwrite(name, frame)
    n_frame += 1
video.release()

print('Done and images are stored in %s' % img_root)