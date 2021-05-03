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

## [Important, read me first] Large Update by AG on 4/5/2021

### Representation Learning
In the middle of UNet and discriminator, there is an additional section to enable unsupervised representation disentanglement. This function can be switched on/off by the program caller, with the `-a` or `--num_attr` argument.

```
python train.py photo2cartoon --resize 256 --batch_size 4 --num_attr 5
```
The above command creates 5 latent variables for representation learning.

### Evaluation on Representation Disentanglement
You may call `eval_rep.py` for any tests related to labelling attributes. This script has 2 different modes:

1. Attribute Scoring \
   This is enabled by default. It allows you to read **a single image** and print its classified attributes, according to how it was trained. \
   It has the positional arguments as follows:
   ```
   py eval_rep.py <image file> <A/B> -r <saved model> --resize <image size during training> --num_attr <number of attributes during training>
   ```
   
   For example,
   ```
   py eval_rep.py dataset/photo2cartoon/testA/000001.jpg A -r saved/photo2cartoon_epoch_20.pth --resize 128 --num_attr 5
   ```

   The output will contain the translated image and the values of the attributes respectively. Note that here, the invoked number must be equal to the number of attributes during training. Otherwise the model will fail to load.
   
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
   And `example_attr.csv` contains a single rows of continuous attribute values, in CSV format. For example, argument with `-a 5` should have 5 values:
   ```
   0,0,0,0,0
   ```