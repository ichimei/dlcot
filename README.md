# dlcot

Deep learning cotton boll segmentation project, based on the [PyTorch DeepLab v3+ implementation](https://github.com/jfzhang95/pytorch-deeplab-xception) by @jfzhang95.

**Last Updated: Apr 10, 2021**

# License

[The MIT License](LICENSE)

# Introduction

This is part of the cotton picking robot project. Given a 2D image from the camera, this module does image segmentation and returns a mask that tells which of the pixels are cotton (white, #FFFFFF) and which are the background (black, #000000).

Our solution is to take photos in the cotton field, label manually, and train using a deep learning model called DeepLab v3+. We directly use the PyTorch solution by @jfzhang95 and write some more scripts to make it easy to use.

Please also read the [README](README_old.md) by the original author.

# Installation

The code has been tested with Anaconda and Python 3.8.

Install [Anaconda](https://www.anaconda.com) and [PyTorch](https://pytorch.org).

For other dependencies,
```sh
pip install matplotlib opencv-python pillow tensorboardX tqdm
```

# Usage

To use this module, you should
1. Create a dataset
2. Train the model using the dataset
3. Test the model given an image

## Create a dataset

To create a dataset, you should have already taken photos in the cotton field and labeled them manually, so you have the original images (in `.jpg`) and mask images (`.png`). The mask images should only have two colors: white (#FFFFFF, for cotton) and black (#000000, for background). The file names should match, for example, the original image `1.jpg` has the mask `1.png`.

### Label the images

To label the images, we recommend two tools.

#### Labelme

[Labelme](https://github.com/wkentaro/labelme) is an open-source graphical image labeling tool. You can draw polygons on an image, and this tool saves the labels in a JSON file. The polygons are saved as lists of coordinates. Click on the link to see the installation and usage.

We provide [`dump_labelme.py`](dump_labelme.py) which can convert (dump) such a JSON file to a mask. It depends on `labelme` (you might have installed) and `imagesize`.
```sh
pip install labelme imagesize
```
For an image `a.jpg` with its label `a.json`, run
```sh
python dump_labelme.py a.json
```
It will automatically search for `a.jpg` (required for getting image size) in the same directory and output the mask to `a.png`. The image name and the mask name are processed automatically and cannot be set manually. (You don't want to do that, right?)

You can also pass multiple filenames. The program iterates over all the arguments and processes each file. You can run like this:
```sh
python dump_labelme.py a.json b.json c.json
python dump_labelme.py *.json    # Using wildcard
```

#### Labelbox

[Labelbox](https://labelbox.com) is an online graphical image labeling tool. You may upload the images, label online and export. Not only polygons, you may also let the system segment the image into superpixels and simply select superpixels to label.

Labelbox exports the labels of a whole dataset to one JSON file. Unlike labelme, the JSON file contains URLs of mask images (in `.png`), and you do not need to convert from polygon coordinates, etc.

However, if you label the image as more than one target (e.g. two cotton bolls are not labeled as the same target, but two separate targets in the system), each target will result in a separate image. We provide [`dump_labelbox.py`](dump_labelbox.py) which can automatically download mask images and merge the targets.

For an exported JSON file `a.json`, run
```sh
python dump_labelbox.py output/
```
to save the `.png` mask images to `output/`. The image names are already contained in the JSON, so no need to specify.

**Important note:** Sometimes the mask images given by Labelbox are turned 180 degrees. This might be caused by a bug when loading the image during labeling. To fix this, we provide [`turn_180.py`](turn_180.py). It takes one or more mask images (in fact it takes effect on any images) and turns each of them 180 degrees respectively.
```sh
python turn_180.py a.png b.png c.png
```

### Directory hierarchy

After labeling, create a directory somewhere (e.g. `cotsh/`), create four subdirectories, and organize your files like this:
- `cotsh/orig/`: Put all the original images (in `.jpg`) here.
- `cotsh/mask_old/`: Put all the mask images (in `.png`) here.
- `cotsh/mask/`: Put all the normalized mask images (in `.png`) here.
- `cotsh/sets/`: This includes two text files which tells the training and validation set.

### For `cotsh/mask/`

What is *normalized mask images*? The masks are in two colors, #FFFFFF and #000000. This is not clear enough - we want to tell the program that #FFFFFF is 1 and #000000 is 0. To do this, we convert the mask to another PNG where each pixel has only one channel (yes, a PNG file could do this), and the value should only be 0 and 1.

You can use [`norm_mask.py`](norm_mask.py) in this repository to normalize the masks.
```sh
python norm_mask.py -m cotsh/mask_old/ -n cotsh/mask/
```
This automatically finds masks in `cotsh/mask_old/` and saves normalized ones in `cotsh/mask/`.

### For `cotsh/sets/`

In this directory, you should tell how the images should be splitted into training and validation set. We do not include the test set in the dataset, because it is not useful in training. In this directory, create two files:
- `train.txt`: Tell the image names which should be treated as a training set.
- `val.txt`: Tell the image names which should be treated as a validation set.

In either text file, the names should *not* include the extension `.jpg` or `png`.

For example, imagine you have three images with names `1.jpg`, `2.jpg`, `3.jpg`. To make the first two images the training set and the third image the validation set, the `train.txt` should look like:
```
1
2
```
and the `val.txt` should look like:
```
3
```
You can use [`split_train_val.py`](split_train_val.py) in this repository to make the splitting easier. The following command takes the image names in `cotsh/orig/`, randomly selects `200` images as the training set and `23` images as the validation set. Then the program outputs to `cotsh/sets/train.txt` and `cotsh/sets/val.txt`.
```sh
python split_train_val.py 200 23 -f cotsh/orig/ -t cotsh/sets/train.txt -v cotsh/sets/val.txt -r
```
The directory `cotsh/orig/` should contain no less than 200 + 23 = 223 images, or an exception would be raised. If containing more than this number, only the given number of images would be selected, and the rest would be ignored (you can use them as test set if you like).

If the `-r` option is given, the select would be performed randomly. Otherwise, the program sorts the filenames, and take the first 200 images as the training set, and the next 23 images as the validation set.

## Train the model

Edit [`mypath.py`](mypath.py). We edit the files based on the `pascal` dataset, so please edit line 5 to the path (e.g. `cotsh/`) of the dataset we have just done.
```python
...
        if dataset == 'pascal':
            return 'cotsh/'  # Edit here!
...
```
Run [`train.py`](train.py) to train the model using the dataset named `pascal`.
```sh
python train.py --backbone resnet --lr 0.007 --workers 2 --epochs 50 --eval-interval 1 --dataset pascal --directory run/ --gpu-ids 0,1
```
To understand the options, see the help.
```sh
python train.py -h
```

For each experiment (i.e. each time you run), the program saves the checkpoints at `run/experiment_x/`. For the first time, it would be `run/experiment_0/`.

The checkpoint name looks like `ckpt_epoch_xxxx.pth`. The program saves a checkpoint every 5 epoches. The result is evaluated on the validation set according to the mean intersection over union (aka. mIoU). For a checkpoint which has the best mIoU in the current experiment, the filename ends with `_best.pth`.

If the program encounters a checkpoint which has the best mIoU in all the experiments, the filename and mIoU value will be included in `run/best_ever.txt`.

**Important note:** you should set `PYTHONPATH` to the *parent* directory of this repository before running this. This is because the whole `dlcot` repository is as a submodule of a larger robot project, and we use `dlcot.xxx` in all the files when importing other files in the same repo. In fact, the original author does not do this, and we add the `dlcot.` in all the files manually. The original version does not require setting `PYTHONPATH`.

We have already done this for you in `train.sh`. You may also set the `PYTHONPATH` in `~/.bashrc` if you like.

## Test the model

After training, you can pick one `.pth` file for testing. Then run [`test.py`](test.py). For a model file (i.e. checkpoint file) `my.pth`, an input `input.jpg` results in an output `output.png`.
```sh
python test.py --in-path input.jpg --out-path output.png --model my.pth --gpu 0
```
To understand the options, see the help.
```sh
python test.py -h
```
Note that you may only use one GPU because we have tested that it would be slower in a multi-GPU design. In addition, also mind the `PYTHONPATH` here.

## Use [`test.py`](test.py) as a module

As mentioned, we use this in a cotton picking robot project. The `dlcot` repo is used [here](https://github.com/houjiawei11/cotton_seg_ros/tree/master/cotton_srv/scripts). You can see that it is actually used as a git submodule.

To use [`test.py`](test.py) as a module, you should run your Python script under the *parent directory* of `dlcot`. Simply import it.
```python
import dlcot.test as dltest
```
Then load the model file in `.pth`. The model should be loaded only once.
```python
model = dltest.load_model('my.pth')
```
When you want to segment an image, input the `model` and a `numpy` RGB `image` (if you use `cv2`, note that you should convert the image to RGB instead of BGR):
```python
mask = dltest.segment(image, model)
```
The `mask` is already in 0 and 255 although the training data is 0 and 1. You can see the implementation in `test.py`.
