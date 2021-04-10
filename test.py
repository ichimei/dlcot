from collections import OrderedDict
import argparse
import os
import time
import cv2
import numpy as np

import torch
torch.set_grad_enabled(False)
CUDA = torch.cuda.is_available()

from torchvision import transforms as tr
from dlcot.modeling.deeplab import DeepLab

def transform(image):
    return tr.Compose([
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

def strip_state(state_dict):
    res = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        res[k] = v
    return res

def load_model(model_file):
    map_location = 'cpu' if not CUDA else None
    m = torch.load(model_file, map_location=map_location)
    state_dict = m['state_dict']
    state_dict = strip_state(state_dict)
    model = DeepLab(num_classes=m['num_classes'],
                    backbone=m['backbone'],
                    output_stride=m['output_stride'],
                    sync_bn=m['sync_bn'],
                    freeze_bn=m['freeze_bn'])
    model.load_state_dict(state_dict)
    model.eval()
    if CUDA:
        model.cuda()
    return model

def segment(np_image, model):
    x = time.time()
    image = transform(np_image).unsqueeze(0)
    if CUDA:
        image = image.cuda()
    output = model(image)
    output = output.cpu().squeeze().detach().numpy()
    mask = np.argmax(output, axis=0)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    print('Time:', time.time() - x)
    print('Size:', mask.shape)
    return mask

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('-i', '--in-path', type=str, required=True, help='image to test')
    parser.add_argument('-o', '--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='saved model path')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA')
    parser.add_argument('--gpu', type=int, default=0,
                        help='sets GPU ID')
    args = parser.parse_args()
    if args.no_cuda:
        global CUDA
        CUDA = False
    elif CUDA:
        torch.cuda.set_device(args.gpu)

    model = load_model(args.model)
    image = cv2.imread(args.in_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred = segment(image, model)
    cv2.imwrite(args.out_path, pred)

if __name__ == "__main__":
    main()
