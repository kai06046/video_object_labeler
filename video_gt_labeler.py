from os.path import join, isdir
from os import makedirs
import os
import sys
import argparse
import json
import numpy as np
import torch

import cv2
import time as time
from util import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, bbox_2_cxy_wh
from util import empty_record, empty_box, empty_point
from util import randomString
from net import DCFNet
from copy import deepcopy

parser = argparse.ArgumentParser(description='Groundtruth labeler for video with DCFNet')
parser.add_argument('--video', help='path of video that to be labeled')
parser.add_argument('--id', default="0", help='the ID of target')
parser.add_argument('--replay_only', default="0", help="replay results only")
parser.add_argument('--model', metavar='PATH', default='param.pth')
parser.add_argument('--n', default=1, type=int, help='start labeling from n frame')
parser.add_argument('--flip', default=1, type=int, help='if 1 then flip the input video')
args = parser.parse_args()

class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    feature_path = 'param.pth'
    crop_sz = 125

    lambda0 = 1e-4
    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    num_scale = 3
    scale_step = 1.0275
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(), signal_ndim=2)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()

video_path = args.video
root = "/".join(video_path.split('/')[:-1])
video_name = video_path.split('/')[-1].split('.')[0]
image_path = join(root, video_name)
print(image_path)
result_path = join(root, video_name + "_" + args.id + '.txt')
res = []

video = cv2.VideoCapture(video_path)
if not os.path.isdir(image_path):
    print("Image files not found, run converting scripts automatically...")
    os.system("python video2img.py --video %s --flip %s" % (video_path, args.flip))
n_total_frames = len(os.listdir(image_path))
cv2_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
if n_total_frames != cv2_frame_count:
    print("Warning: number of images (%d) and frame count of cv2 (%d) are different" % (n_total_frames, cv2_frame_count))
print("There are total %d frames in this video" % n_total_frames)

if args.flip == 1:
    image_w = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    image_h = video.get(cv2.CAP_PROP_FRAME_WIDTH)
else:
    image_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    image_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

half_image_w = int(image_w/2)
half_image_h = int(image_h/2)
target_pos = target_sz= None
patch = patch_crop = None
n_frame = args.n-1
is_visible = 1
is_labeling = False

start_from_half = False

if n_frame != 0:
    is_result_path = os.path.isfile(result_path)
    if not is_result_path:
        print("Warning: Can't find %s.\nStart labeling at the first frame." % (result_path))
        n_frame = 0
    else:
        res = np.loadtxt(result_path, delimiter=',')
        res[:, 0] *= image_w if image_w < 1080 else int(image_w/2)
        res[:, 2] *= image_w if image_w < 1080 else int(image_w/2)
        res[:, 1] *= image_h if image_w < 1080 else int(image_h/2)
        res[:, 3] *= image_h if image_w < 1080 else int(image_h/2)
        res = res.tolist()

        length_res = len(res)
        # print("length of the file: ",length_res)
        if args.n > length_res:
            print("Warning: There are only %d records instead of %d. Start labeling at %d frame." % (length_res, args.n, length_res))
            n_frame = length_res - 1
        xmin, ymin, xmax, ymax, is_visible = res[n_frame]

        if is_visible == 1:
            x_c = (xmin + xmax)/ 2.0
            y_c = (ymin + ymax)/ 2.0
            w = (xmax - xmin)
            h = (ymax - ymin)
            target_pos = np.array([x_c, y_c])
            target_sz = np.array([w, h])

        start_from_half = True

# default parameter and load feature extractor network
config = TrackerConfig()
net = DCFNet(config)
net.load_param(args.model)
net.eval().cuda()

def update_template():
    # confine results
    min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
    max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

    window_sz = target_sz * (1 + config.padding)
    bbox = cxy_wh_2_bbox(target_pos, window_sz)
    patch = crop_chw(im, bbox, config.crop_sz)

    target = patch - config.net_average_image
    net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

    patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)

def draw():
    global im_show, target_pos, target_sz, res, im
    im_show = im.copy()
    
    if target_pos is not None:
        cv2.rectangle(im_show, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                      (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                      (0, 255, 0), 1)
    cv2.putText(im_show, str(n_frame + 1), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

def save_results(res, result_path, image_w, image_h):
    res = np.array(res).copy()
    # normalized
    res[:, 0] /= image_w if image_w < 1080 else int(image_w/2)
    res[:, 2] /= image_w if image_w < 1080 else int(image_w/2)
    res[:, 1] /= image_h if image_w < 1080 else int(image_h/2)
    res[:, 3] /= image_h if image_w < 1080 else int(image_h/2)

    with open(result_path, 'w') as f:
        for x in res:
            f.write(','.join([str(i) for i in x]) + '\n')

KEY_LEFT = ord('a')
KEY_RIGTH = ord('d')
KEY_DELETE = ord('w')
KEY_PAUSE = ord(' ')

speed = []
tmp_p_1 = (0, 0)
tmp_p_2 = (0, 0)
n_reset = 0

# mouse callback (retarget the bounding box)
def mouse_opt(event, x, y, flags, param):
    global tmp_p_1, tmp_p_2, target_pos, target_sz, n_reset, is_visible, res, is_labeling
    if event == cv2.EVENT_LBUTTONDOWN:

        if not is_labeling:
            tmp_p_1 = (min(max(0, x), half_image_w if image_w > 1080 else image_w), min(max(0, y), half_image_h if image_w > 1080 else image_h))
            print("mouse on click at (%d, %d)" % (x, y))
            is_labeling = True
        else:
        # tmp_p_1 = (x, y)
    # elif event == cv2.EVENT_LBUTTONUP:
            tmp_p_2 = (min(max(0, x), half_image_w if image_w > 1080 else image_w), min(max(0, y), half_image_h if image_w > 1080 else image_h))
            # tmp_p_2 = (x, y)
            n_reset += 1

            # update target bounding box
            x_c = (tmp_p_1[0] + tmp_p_2[0]) / 2.0
            y_c = (tmp_p_1[1] + tmp_p_2[1]) / 2.0
            w = float(abs(tmp_p_2[0] - tmp_p_1[0]))
            h = float(abs(tmp_p_2[1] - tmp_p_1[1]))
            target_pos = np.array([x_c, y_c])
            target_sz = np.array([w, h])
            # try:
            # res[n_frame] = np.concatenate((cxy_wh_2_bbox(target_pos, target_sz), np.array([is_visible])))
            # except Exception as e:
            #     print(e)
            #     save_results(res, result_path, image_w, image_h)
            #     print("save file in %s" % result_path)
            #     sys.exit()
            if len(res) == 0:
                print("no record found, append data")
                res.append(np.concatenate((cxy_wh_2_bbox(target_pos, target_sz), np.array([is_visible]))))            
            else:
                res[n_frame-1] = np.concatenate((cxy_wh_2_bbox(target_pos, target_sz), np.array([is_visible])))
            # update template
            update_template()

            is_visible = 1
            is_labeling = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_labeling:
            tmp_p_2 = (min(max(0, x), half_image_w if image_w > 1080 else image_w), min(max(0, y), half_image_h if image_w > 1080 else image_h))

            # update target bounding box
            x_c = (tmp_p_1[0] + tmp_p_2[0]) / 2.0
            y_c = (tmp_p_1[1] + tmp_p_2[1]) / 2.0
            w = float(abs(tmp_p_2[0] - tmp_p_1[0]))
            h = float(abs(tmp_p_2[1] - tmp_p_1[1]))
            target_pos = np.array([x_c, y_c])
            target_sz = np.array([w, h])
            is_visible = 1
            draw()
            cv2.imshow('video', im_show)
            cv2.waitKey(1)

# fps = video.get(cv2.CAP_PROP_FPS)
# frame_msec = 1000 / fps
# video_time = 0

cv2.namedWindow("video", 1)
cv2.startWindowThread()

# loop frames, tracking and record bounding box information
if args.replay_only == "0":
    tic = time.time()  # time start 
    while n_frame < n_total_frames:
        file_path = join(image_path, "%06d.jpg" % n_frame)
        if os.path.isfile(file_path):
            im = cv2.imread(file_path)
        else:
            print("Can't load %s..." % file_path)
            pass
        # ok, im = video.read()
        # if not ok:
        #     print('This is not a valid video')
        #     break

        # resize frame size if it is too big
        # if image_w > 1080:
        #     im = cv2.resize(im, (half_image_w, half_image_h))
        im_show = im.copy()

        # loop the first frame to get first frame target bounding box
        if (n_frame == 0 and target_pos is None) or start_from_half:
            while True:
                im_show = im.copy()
                draw()
                cv2.putText(im_show, 'Use left mouse button to draw a bouding box and press space to start', (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('video', im_show)
                cv2.setMouseCallback('video', mouse_opt)
                key_first_frame = cv2.waitKey(1)
                # if cv2.getWindowProperty('video', 1) == -1: # can't work in MobaXterm
                #     sys.exit()

                # press space and first frame bounding box is choosed by user, then continue the video loop
                if key_first_frame == KEY_PAUSE:
                    if target_pos is not None:
                        is_visible = 1
                        # confine results
                        min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
                        max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

                        # crop template
                        window_sz = target_sz * (1 + config.padding)
                        bbox = cxy_wh_2_bbox(target_pos, window_sz)
                        patch = crop_chw(im, bbox, config.crop_sz)

                        target = patch - config.net_average_image
                        net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

                        patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
                    else:
                        is_visible = 0
                    if len(res) > 0:
                        res = res[:n_frame]
                    res.append(np.concatenate((cxy_wh_2_bbox(target_pos, target_sz), np.array([is_visible]))))  # save in .txt
                    start_from_half = False
                    break
                # previous frame
                elif key_first_frame == KEY_LEFT:
                    if n_frame > 0:
                        n_frame -= 1
                        target_pos, target_sz = bbox_2_cxy_wh(res[n_frame])
                        im = cv2.imread(join(image_path, "%06d.jpg" % n_frame))
                        im_show = im.copy()
                        draw()
                # next frame
                elif key_first_frame == KEY_RIGTH:
                    if n_frame < n_total_frames and n_frame < (len(res)-1):
                        n_frame +=1
                        target_pos, target_sz = bbox_2_cxy_wh(res[n_frame])
                        im = cv2.imread(join(image_path, "%06d.jpg" % n_frame))
                        im_show = im.copy()
                        draw()
                # press r to reset frame size (just in case sometimes MobaXterm doesn't work well...)
                elif key_first_frame == ord('r'):
                    cv2.destroyAllWindows()
                elif key_first_frame == 27:
                    print('Exit the program')
                    sys.exit()
        elif im is None:
            print('Image is None at frame: %d \n Something wrong, please check.' % n_frame)
            break
        else:
            try:
                if is_visible == 1 and target_pos is not None:
                    # init patch
                    if patch is None and patch_crop is None:
                        min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
                        max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

                        window_sz = target_sz * (1 + config.padding)
                        bbox = cxy_wh_2_bbox(target_pos, window_sz)
                        patch = crop_chw(im, bbox, config.crop_sz)

                        target = patch - config.net_average_image
                        net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

                        patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
                    else:
                        ################################### tracking algo ###################################
                        for i in range(config.num_scale):  # crop multi-scale search region
                            window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
                            bbox = cxy_wh_2_bbox(target_pos, window_sz)
                            patch_crop[i, :] = crop_chw(im, bbox, config.crop_sz)

                        search = patch_crop - config.net_average_image
                        response = net(torch.Tensor(search).cuda())
                        peak, idx = torch.max(response.view(config.num_scale, -1), 1)
                        peak = peak.data.cpu().numpy() * config.scale_penalties
                        best_scale = np.argmax(peak)
                        r_max, c_max = np.unravel_index(idx[best_scale].cpu().numpy(), config.net_input_size)

                        if r_max > config.net_input_size[0] / 2:
                            r_max = r_max - config.net_input_size[0]
                        if c_max > config.net_input_size[1] / 2:
                            c_max = c_max - config.net_input_size[1]
                        window_sz = target_sz * (config.scale_factor[best_scale] * (1 + config.padding))

                        target_pos = target_pos + np.array([c_max, r_max]) * window_sz / config.net_input_size
                        target_sz = np.minimum(np.maximum(window_sz / (1 + config.padding), min_sz), max_sz)

                        # model update
                        window_sz = target_sz * (1 + config.padding)
                        bbox = cxy_wh_2_bbox(target_pos, window_sz)
                        patch = crop_chw(im, bbox, config.crop_sz)
                        target = patch - config.net_average_image
                        net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=config.interp_factor)
                        ################################### end ###################################
                elif is_visible == 1 and target_pos is None:
                    print('Warning: there is conflict at frame: %d (length of results: %d)' % (n_frame, len(res)))
                    print('Target pos: ', target_pos, "  Target size: ", target_sz)
                else:
                    # is visible = 0, pass
                    print("sss")
                    pass
                draw()
                cv2.imshow('video', im_show)
                key = cv2.waitKey(1)
                # press space to pause the tracking and wait for other keyboard/mouse callbacks
                if key == KEY_PAUSE:

                    while True:
                        cv2.imshow('video', im_show)
                        key2 = cv2.waitKey(0)
                        # press space to continue tracking
                        if key2 == KEY_PAUSE:
                            if (n_frame+1) < len(res):
                                # update results
                                res = res[:n_frame]
                            if target_pos is None:
                                is_visible = 0
                            else:
                                # target_pos, target_sz = bbox_2_cxy_wh(res[-1])
                                is_visible = 1
                            break
                        # previous frame
                        elif key2 == KEY_LEFT:
                            if n_frame > 0:
                                n_frame -= 1
                                target_pos, target_sz = bbox_2_cxy_wh(res[n_frame])
                                im = cv2.imread(join(image_path, "%06d.jpg" % n_frame))
                                im_show = im.copy()
                                draw()
                        # next frame
                        elif key2 == KEY_RIGTH:
                            if n_frame < n_total_frames and n_frame < (len(res)-1):
                                n_frame +=1
                                target_pos, target_sz = bbox_2_cxy_wh(res[n_frame])
                                im = cv2.imread(join(image_path, "%06d.jpg" % n_frame))
                                im_show = im.copy()
                                draw()
                        # delete bounding box
                        elif key2 == KEY_DELETE:
                            is_visible = 0
                            target_pos = target_sz = None
                            draw()
                        elif key2 == 27:
                            save_results(res, result_path, image_w, image_h)
                            print('Exit the program')
                            sys.exit()

                elif key in [27, 1048603]:
                    print('Exit labeling')
                    break
                cv2.setMouseCallback('video', mouse_opt)
            except Exception as e:
                print("n frame: ", n_frame, " Error: ", e)
            # append bounding box results
            res.append(np.concatenate((cxy_wh_2_bbox(target_pos, target_sz), np.array([is_visible]))))
        # print('Current frame: %d' % (n_frame+1))
        # if target_pos is not None:
           #  print('Bounding box: ', (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
           #                            (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)))
        n_frame += 1        

    toc = time.time() - tic
    fps = n_frame / toc
    speed.append(fps)
    print('Done!\nTime: {:3.1f}s\tSpeed: {:3.1f}fps'.format(toc, fps))

    # save result
    save_results(res, result_path, image_w, image_h)
    video.release()

    print('***Total Mean Speed: {:3.1f} (FPS)  Total number of user retarget: {:d} ***'.format(np.mean(speed), n_reset))

    cv2.destroyAllWindows()
#------------------------------------------------------ end labeling ------------------------------------------------------#

# play back results and save video
print('Relaying the results')
video = cv2.VideoCapture(video_path)
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out_root = os.path.join("./Tracking", "labeling_results", video_name)
if not os.path.isdir(out_root):
    os.makedirs(out_root, exist_ok=True)
out = cv2.VideoWriter(os.path.join(out_root, video_name+ "_" + args.id + ".avi"), cv2.VideoWriter_fourcc(*'XVID'), 30.0, (int(image_w) if image_w < 1080 else int(image_w/2),int(image_h) if image_w < 1080 else int(image_h/2)))

res = np.loadtxt(result_path, delimiter=',')
image_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
image_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

n_frame = 0

def write_json(res, file_path, image_w, image_h):

    r = deepcopy(empty_record)
    r['asset']['id'] = randomString(32)
    r['asset']['name'] = os.path.basename(file_path)
    r['asset']['path'] = os.path.abspath(file_path)
    r['asset']['size']['height'] = image_h
    r['asset']['size']['width'] = image_w

    xmin, ymin, xmax, ymax, vis = res
    if vis == 1:
        xmin = xmin * image_w
        ymin = ymin * image_h
        xmax = xmax * image_w
        ymax = ymax * image_h
    else:
        xmin = ymin = xmax = ymax = 0.0

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)

    b = deepcopy(empty_box)
    b['id'] = randomString(9)
    b['boundingBox']['height'] = h
    b['boundingBox']['left'] = xmin
    b['boundingBox']['top'] = ymin
    b['boundingBox']['width'] = w

    p1 = deepcopy(empty_point)
    p2 = deepcopy(empty_point)
    p3 = deepcopy(empty_point)
    p4 = deepcopy(empty_point)

    p1['x'] = xmin
    p1['y'] = ymin
    p2['x'] = xmax
    p2['y'] = ymin
    p3['x'] = xmax
    p3['y'] = ymax
    p4['x'] = xmin
    p4['y'] = ymax
    b['points'].extend([p1, p2, p3, p4])
    r['regions'].append(b)

    out_dir = os.path.dirname(file_path) + "_json"
    if not os.path.isdir(out_dir):
        print("folder doesn't exist, create one.")
        os.makedirs(out_dir)
    out_file_path = os.path.join(out_dir, os.path.basename(os.path.dirname(file_path)) + "_" + os.path.basename(file_path).replace("jpg", "json"))

    with open(out_file_path, 'w') as out:
        json.dump(r, out)

while n_frame < n_total_frames:
    file_path = join(image_path, "%06d.jpg" % n_frame)
    frame = cv2.imread(file_path)

    image_w = frame.shape[1]
    image_h = frame.shape[0]

    write_json(res[n_frame], file_path, image_w, image_h)

    if n_frame < len(res):
        xmin, ymin, xmax, ymax, vis = res[n_frame]
        if vis == 1:
            xmin = int(xmin * image_w)
            ymin = int(ymin * image_h)
            xmax = int(xmax * image_w)
            ymax = int(ymax * image_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3, 1)

    if image_w > 1080:
        frame = cv2.resize(frame, (int(image_w/2), int(image_h/2)))

    cv2.putText(frame, str(n_frame + 1), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    k = cv2.waitKey(1)
    cv2.imshow('valid', frame)
    out.write(frame)
    
    if k == 27:
        break
    elif k == KEY_PAUSE:
        while True:
            cv2.imshow('valid', frame)
            k2 = cv2.waitKey(0)
            if k2 == ord(' '):
                break
    n_frame += 1
cv2.destroyAllWindows()
video.release()
out.release()
