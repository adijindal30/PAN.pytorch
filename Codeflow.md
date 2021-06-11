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
The bounding box should have the coordinate in the following manner: x1, y1, x2, y2, x3, y3, x4, y4,a.

In preprocessing step, for our custom Tote-Id dataset, we divide the image into 4 subimages(with 100 pixels overlap), and so we divide the bounding box correspondingly.

### train: 
prepare a <em>dataset.txt</em> file in the following format, use '\t' as a separator
```bash
/path/to/img_1.jpg path/to/label_1.txt
/path/to/img_2.jpg path/to/label_2.txt
/path/to/img_3.jpg path/to/label_3.txt
...
```


### val:
use a folder contains
```bash
img/ store img
gt/ store gt file
```


## Train
For more information refer **train_flow.md**.
1. config the `train_data_path/dataset.txt`,`val_data_path`(contains img and gt folder)in [config.json](config.json)
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


| Backbone Model       | Train dataset | Image Dimension | Epoch | Batch Size | Pretrained weights| Test dataset   | F1       | Precision | Recall   | FPS   |
| -------------------- | ---------------------------------------- | --------------- | ----- | ---------- | ------------------------------------ | ----------------- | -------- | --------- | -------- | ----- |
| Mobilenetv2/FPEM     | ICDAR-1000                               | 640             | 45    | 4          | yes/Imagenet                         | ICDAR-500(Test)   | 0.19     | 0.38      | 0.13     | 11    |
| Renset 18            | ICDAR-1000                               | 640             | 200+  | 4          | yes/Imagenet                         | ICDAR-500(Test)   | 0.78     | 0.85      | 0.73     | 13.25 |
| shufflenetv2         | ICDAR-1000                               | 640             | 200+  | 4          | yes/Imagenet                         | ICDAR-500(Test)   | 0.72     | 0.79      | 0.67     | 16.84 |
| densenet121          | ICDAR-1000                               | 512             | 40+   | 4          | yes/Imagenet                         | ICDAR-500(Test)   | 0.5      | 0.619     | 0.402    | 3     |
| squeezenet1\_1       | ICDAR-250                                | 512             | <20   | 4          | No                                   | ICDAR-500(Test)   | \-       |           |          | 6     |
| shufflenetv2(fpem=1) | ICDAR-250                                | 512             | <20   | 4          | yes/Imagenet                         | ICDAR-500(Test)   | \-       |           |          | 19-20 |
| resnet18(fpem=1)     | ICDAR-250                                | 640             | <20   | 4          | yes/Imagenet                         | ICDAR-500(Test)   |          |           |          | 16.87 |
| Mobilenetv2/FPEM=1   | ICDAR-1000                               | 640             | 45    | 4          | yes/Imagenet                         | ICDAR-500(Test)   | 0.458438 | 0.66      | 0.34     | 15    |
| squeezenet1\_0       | ICDAR-250                                | 512             |       | 4          | yes/Imagenet                         |                   |          |           |          |       |
| Resnetxt             |                                          |                 |       | 4          |                                      |                   |          |           |          |       |


### [Tote-Id dataset]()
| Backbone Model       | Train dataset | Image Dimension | Epoch | Batch Size | Pretrained weights| Test dataset   | F1       | Precision | Recall   | FPS   |
| -------------------- | ---------------------------------------- | --------------- | ----- | ---------- | ------------------------------------ | ----------------- | -------- | --------- | -------- | ----- |
| shufflenetv2(fpem=0) | Tote-id Datset-1058                      | 640             | 15    | 4          | Yes/Imagenet                         | Tote-id Datset-56 | 0.325517 | 0.442362  | 0.257501 | 9     |

### examples

![](imgs/example/img_29.jpg)

![](imgs/example/img_75.jpg)


### reference
1. https://arxiv.org/pdf/1908.05900.pdf
2. https://github.com/WenmuZhou/PSENet.pytorch
3. https://pytorch.org/vision/stable/models.html
4. https://github.com/WenmuZhou/PAN.pytorch

