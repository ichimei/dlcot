import sys
import json
import cv2
import numpy as np
from labelme.utils.shape import shape_to_mask
import imagesize

def load_json(fjs):
    with open(fjs) as f:
        js = f.read()
    return json.loads(js)

def dump(js, name):
    base = name.rsplit('.', maxsplit=1)[0]
    w, h = imagesize.get(base + '.jpg')
    m = np.zeros((h, w), dtype=bool)
    shapes = js['shapes']
    for shape in shapes:
        points = shape['points']
        n = shape_to_mask((h, w), points)
        m = np.logical_or(m, n)
    res = m.astype(np.uint8) * 255
    cv2.imwrite(base + '.png', res)

if __name__ == '__main__':
    fjss = sys.argv[1:]
    for fjs in fjss:
        print(f'[INFO] Processing {fjs}...')
        if fjs.endswith('.json'):
            js = load_json(fjs)
            dump(js, fjs)
