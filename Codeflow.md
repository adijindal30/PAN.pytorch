# Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network

<!-- ![](imgs/paper/PAN.jpg) -->

## Requirements
* pytorch 1.4
* torchvision 0.5
* pyclipper
* opencv3
* gcc 4.9+

## Dataset 

- ICDAR-2015 dataset [(download link)](https://rrc.cvc.uab.es/?ch=4&com=downloads)
  - Train size: 1000 images, Test size: 500 images
  - Dimension: 720 X 1280 X 3
- Tote-Id dataset
  - Train size: 300 images, Test size: 76 images
  - Dimension:  1440 X 1920  X 3

## Data Preparation and preprocessing

To perform real scene text detection, we need good quality images as input and corresponding bounding box of the texts as output(text as well, if one need text recognition).\

For our custom Tote-Id dataset, we divide our image into 4 subimages 

### train: 
prepare a text in the following format, use '\t' as a separator
```bash
/path/to/img.jpg path/to/label.txt
...
```
### val:
use a folder
```bash
img/ store img
gt/ store gt file
```

## Train
For more information refer **train_flow.md**.
1. config the `train_data_path`,`val_data_path`in [config.json](config.json)
2. use following script to run
```sh
python3 train.py
```

## Test

[eval.py](eval.py) is used to test model on test dataset

1. config `model_path`, `img_path`, `gt_path`, `save_path` in [eval.py](eval.py)
2. use following script to test
```sh
python3 eval.py
```

This will produce the inference speed(in FPS) and corresponding precision, recall & F1-score.

## Predict 
[predict.py](predict.py) is used to inference on single image

1. config `model_path`, `img_path`, in [predict.py](predict.py)
2. use following script to predict
```sh
python3 predict.py
```


<h2 id="Performance">Performance</h2>

### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4)
only train on ICDAR2015 dataset

| Method                   | image size (short size) |learning rate | Precision (%) | Recall (%) | F-measure (%) | FPS |
|:--------------------------:|:-------:|:--------:|:--------:|:------------:|:---------------:|:-----:|
| paper(resnet18)  | 736 |x | x | x | 80.4 | 26.1 |
| my (ShuffleNetV2+FPEM_FFM+pse扩张)  |736 |1e-3| 81.72 | 66.73 | 73.47 | 24.71 (P100)|
| my (resnet18+FPEM_FFM+pse扩张)  |736 |1e-3| 84.93 | 74.09 | 79.14 | 21.31 (P100)|
| my (resnet50+FPEM_FFM+pse扩张)  |736 |1e-3| 84.23 | 76.12 | 79.96 | 14.22 (P100)|
| my (ShuffleNetV2+FPEM_FFM+pse扩张)  |736 |1e-4| 75.14 | 57.34 | 65.04 | 24.71 (P100)|
| my (resnet18+FPEM_FFM+pse扩张)  |736 |1e-4| 83.89 | 69.23 | 75.86 | 21.31 (P100)|
| my (resnet50+FPEM_FFM+pse扩张)  |736 |1e-4| 85.29 | 75.1 | 79.87 | 14.22 (P100)|
| my (resnet18+FPN+pse扩张)  | 736 |1e-3|  76.50 | 74.70 | 75.59 | 14.47 (P100)|
| my (resnet50+FPN+pse扩张)  | 736 |1e-3|  71.82 | 75.73 | 73.72 | 10.67 (P100)|
| my (resnet18+FPN+pse扩张)  | 736 |1e-4|  74.19 | 72.34 | 73.25 | 14.47 (P100)|
| my (resnet50+FPN+pse扩张)  | 736 |1e-4|  78.96 | 76.27 | 77.59 | 10.67 (P100)|

### examples

![](imgs/example/img_29.jpg)

![](imgs/example/img_75.jpg)


### reference
1. https://arxiv.org/pdf/1908.05900.pdf
2. https://github.com/WenmuZhou/PSENet.pytorch

