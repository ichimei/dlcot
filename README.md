# dlcot

Deep learning cotton boll segmentation project, based on the [PyTorch DeepLab v3+ implementation](https://github.com/jfzhang95/pytorch-deeplab-xception) by @jfzhang95.

**Last Updated: Mar 17, 2021**

# License

[The MIT License](LICENSE)

# Introduction

This is part of the cotton picking robot project. Given a 2D image from the camera, this module does image segmentation and returns a mask that tells which of the pixels are cotton (white, #FFFFFF) and which are the background (black, #000000).

Our solution is to take photos in the cotton field, label manually, and train using a deep learning model called DeepLab v3+. We directly use the @jfzhang95 solution and write some more scripts to make it easy to use.

# Usage

To use this module, you should
1. Create a dataset
2. Train the model using the dataset
3. Test the model given an image

## Create a dataset

To create a dataset, you should have already taken photos in the cotton field and labeled them manually, so you have the original images (in `.jpg`) and mask images (`.png`). The mask images should only have two colors: white (#FFFFFF, for cotton) and black (#000000, for background). The file names should match, for example, the original image `1.jpg` has the mask `1.png`.

Then create a directory somewhere (e.g. `cotsh/`), create four subdirectories, and organize your files like this:
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

Edit `mypath.py`. We edit the files based on the `pascal` dataset, so please edit line 5 to the path (e.g. `./cotsh/`) of the dataset we have just done.
```python
...
        if dataset == 'pascal':
            return 'cotsh/'  # Edit here!
...
```
Train the model using the dataset named `pascal`.
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

After training, you can pick one of the `.pth` files for testing. For a model file (i.e. checkpoint file) `my.pth`, an input `input.jpg` results in an output `output.png`.
```sh
python test.py --in-path input.jpg --out-path output.png --model my.pth --gpu 0
```
To understand the options, see the help.
```sh
python test.py -h
```
Note that you may only use one GPU because we have tested that it would be slower in a multi-GPU design. In addition, also mind the `PYTHONPATH` here.

## Use `test.py` as a module

As mentioned, we use this in a cotton picking robot project. The `dlcot` repo is used [here](https://github.com/houjiawei11/cotton_seg_ros/tree/master/cotton_srv/scripts). You can see that it is actually used as a git submodule.

To use `test.py` as a module, you should run your Python script under the *parent directory* of `dlcot`. Simply import `test.py`.
```python
import dlcot.test as dltest
```
Then load the model file in `.pth`. The model should be loaded only once.
```python
model = dltest.load_model(YOUR_MODEL_PATH)
```
When you want to segment an image, input the `model` and a `numpy` RGB `image` (if you use `cv2`, note that you should convert the image to RGB instead of BGR):
```python
mask = dltest.segment(image, model)
```
The `mask` is already in 0 and 255 although the training data is 0 and 1. You can see the implementation in `test.py`.
