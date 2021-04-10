import sys
import os
import json
import requests
import cv2
import numpy as np

def load_json(fjs):
    with open(fjs) as f:
        js = f.read()
    return json.loads(js)

def dump(js, path):
    for jsone in js:
        name = jsone['External ID']
        print(f'[INFO] Processing {name}...')
        try:
            objs = jsone['Label']['objects']
        except:
            continue
        mask = None
        for i, obj in enumerate(objs):
            link = obj['instanceURI']

            while True:
                try:
                    print(f'  [INFO] Downloading instance {i}...', end='')
                    r = requests.get(link)
                except KeyboardInterrupt:
                    exit()
                except:
                    print(' retrying...', end='')
                else:
                    break

            if r.status_code == 200:
                print(' OK!')
                img = np.frombuffer(r.content, dtype=np.uint8)
                img = cv2.imdecode(img, 0)
                mask = img if mask is None else cv2.bitwise_or(mask, img)
            else:
                print(f'  [ERROR] Status {r.status_code}')

        mName = name.rsplit('.', maxsplit=1)[0] + '.png'
        cv2.imwrite(os.path.join(path, mName), mask)

if __name__ == '__main__':
    fjs = sys.argv[1]
    path = sys.argv[2]
    js = load_json(fjs)
    dump(js, path)
