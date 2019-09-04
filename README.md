# Object labeler for video
This is a labeling tool for tracking single object in a video. 
It requires user to manually choose a bounding box at the first frame of the video. 
Then it will track this target automatically and can be stopped to retarget at anytime.
Current tracking algoritm is based on DCFNet, please reference to this [repo](https://github.com/foolwood/DCFNet_pytorch) for the installation and requirements.

## Usage
First run this script to convert video frames into images
``` bash
python video2img.py --video input.avi
```
Then start labeling with this following script
``` bash
python video_gt_labeler.py --video input.avi --id 0
```

--video: the path of the video that needed labeling 

--id (default=0): the ID of target

--replay_only (default=0): replay existed labeling results

--n (default=1): start labeling from n frame

## Features
The following are some keyboard callbacks to specific function respectively:
* Press key `space`: Pause the tracking/Continue the tracking
    * User can use `left button of mouse` to retarget bounding box while the tracking was paused
* Press key `a` when pause: Go previous frame
* Press key `d` when pause: Go to next frame
* Press key `w` when pause: Label as occlusion and stop running tracking algo
* Press key `Esc`: Exit the tool


## Output
A text file that contains bounding box and visibility (0 means occlusion) information with format like: 
`xmin, ymin, xmax, ymax, is_visible`
that are normalized to the video resolution will be generated in the same path with video after labeling is done.

