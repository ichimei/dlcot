import sys
import cv2
import numpy as np
fs = sys.argv[1:]
for f in fs:
    x = cv2.imread(f, -1)
    x = np.flip(np.flip(x, 0), 1)
    cv2.imwrite(f, x)
