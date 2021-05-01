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

## [Important, read me first] Large Update by AG on 2/5/2021

### Representation Learning (Only available for photo2cartoon now)
In the middle of UNet in generator, there is an additional encoder to transform latent space to attribute space .This function can be switched on/off by the program caller, with the `--attr1` and `--attr2` arguments.

For example, the following call will **train the generators with a ground-truth attribute list, loaded from `list_attr_celeba.csv` and `list_attr_cartoon.csv`**
```
python train.py photo2cartoon --resize 256 --batch_size 4 --attr1 dataset/photo2cartoon/list_attr_celeba.csv --attr2 dataset/photo2cartoon/list_attr_cartoon.csv
```

### Evaluation on Representation Disentanglement
You may call `eval_rep.py` for any tests related to labelling attributes. This script has 2 different modes:

1. Attribute Classification \
   This is enabled by default. It allows you to read **a single image** and print its classified attributes, according to how it was trained. \
   It has the positional arguments as follows:
   ```
   py eval_rep.py <image file> <A/B> -r <saved model> --resize <image size during training>
   ```
   
   For example,
   ```
   py eval_rep.py dataset/photo2cartoon/testA/000001.jpg A -r saved/photo2cartoon_epoch_20.pth --resize 128
   ```
   
2. Generating image with custom attrbute lists \
   Different to the previous mode where the attribute list is generated. This option allows you to **feed a custom attribute list** and observe the generated image. \
   It can be enabled with the `-l` or `--latent` argument.
   ```
   py eval_rep.py <image file> <A/B> -r <saved model> --resize <image size during training> -l <csv file of attribute list>
   ```
   For example,

   ```
   py eval_rep.py dataset/photo2cartoon/trainB/001.png B -r saved/photo2cartoon_epoch_20.pth --resize 128 -l dataset/photo2cartoon/example_attr.csv
   ```
   And `example_attr.csv` contains two rows. The first row is the column names and the second row is the desired attributes to set.
   ```
   Bangs,Blond_Hair,Brown_Hair,Eyeglasses,Mouth_Slightly_Open,Narrow_Eyes,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Necklace
   0,0,1,1,0,0,1,1,0,0,0,0
   ```