# COMP5212 Project - CycleGAN & Representation Disentanglement

This is an adapted implementation of CycleGAN, referenced from [The Original Paper](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

More will be added to this page.

## Usage

First, you may need to download a valid dataset and place into `dataset/<name>`. A valid file structure should contain 4 subfolders, `trainA`, `trainB`, `testA` and `testB`.


Run `train.py` to initiate the training.

### Training with GPU
You can use `--device` to specify the device to load into, and `--batch_size` to control the minibatch size to train with.
```
python train.py --device cuda --batch_size 64 <dataset name> 
```

**<span color='red'>In the current version, all images are reshaped into 64x64.</span>** \
It will be removed in later version. If you wish to remove it in this version, consider using the `--resize` argument.


### Training with CPU (Not recommended)
```
python train.py --device cpu <dataset name> 
```
