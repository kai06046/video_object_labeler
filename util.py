import numpy as np
import cv2
import random
import string

def randomString(stringLength=32):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(stringLength))

empty_record = {
    "asset": {
        "format": 'jpg',
        "id": None,
        "name": None,
        "path": None,
        "size": {
            'height': None,
            'width': None
        },
        "state": 2,
        "type": 1
    },
    "regions": [],
    "version": '2.1.0'
}

empty_box = {
    "id": None,
    "type": 'POLYGON',
    "tags": ["Mykad"],
    "boundingBox": {
        "left": None,
        "top": None,
        "width": None,
        "height": None
    },
    "points": []
}

empty_point = {
    "x": None,
    "y": None
}

def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])  # 1-index

def rect1_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), np.array([rect[2], rect[3]])  # 0-index

def bbox_2_cxy_wh(rect):
    if rect[0] != 0 and rect[2] != 0:
        return np.array([(rect[0]+rect[2])/2, (rect[1]+rect[3])/2]), np.array([rect[2]-rect[0], rect[3]-rect[1]])
    else:
        return None, None

def cxy_wh_2_bbox(cxy, wh):
    if cxy is not None:
        return np.array([cxy[0]-wh[0]/2+1, cxy[1]-wh[1]/2+1, cxy[0]+wh[0]/2+1, cxy[1]+wh[1]/2+1])  # 0-index
    else:
        return np.array([0, 0, 0, 0]).astype(float)


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


if __name__ == '__main__':
    a = gaussian_shaped_labels(10, [5,5])
    print(a)