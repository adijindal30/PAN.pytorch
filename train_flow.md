# Train.py

## Initialization:

### (a) Loading the data:
```bash
train_loader = get_dataloader(parameters)
```
- The parameters are train_path, val_path, val_ratio, input_size etc. Can be changes into config.json\
- This calls data_loader/_init_.py, which further calls data_loader/dataset.py for Imagedatset which augment the data as its resize,crop or horizontally flip 
or can add noise for more generic model.

### (b) Setting up PAN loss:
```bash
criterion = get_loss(config).cuda()
```
- This calls to model/loss.py where PANLoss initialize with hyperparameters as :\
alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean'
- This also setups up training masks, kernel coordinates, text coordinates which use to calculate loss for our text detection model


### (c) Setting up the Deep-Learning model:
```bash
model = get_model(config)
```
The config parameters it have :  "backbone": "shufflenetv2","fpem_repeat": 2,"pretrained": true,"segmentation_head": "FPEM_FFM"\
All parameters are important for getting the best perfomance out of it as :
- backbone: they are weak feature extraction require for segmentation\
**Choices**: 'resnet18','resnet34','resnet50','resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'shufflenetv2','mobilenet_v2', 'densenet121'
- fpem_repeat: means the number of cascaded FPEMs\
**Note**: For each additional FPEM, the FPS will decrease by about 2-5 FPS.
- pretrained: Just for using pretraied weights in backbone model, usually "Imagenet"
- Segmentation_head: \
**Choices**: 
  - "FPEM_FFM"- to enhance the features of different scales by fusing the low-level and high-level information with minimal computationoverhead.
  - "FPN" - Computationally expensive but gives comparable perfomance to FPEM_FFM

This calls models/model.py which sets up the whole model in pytorch framework

### (d) Setting up of optimizer and other checkpoints, etc:\
```bash
trainer = Trainer(config=config,
    model=model,
    criterion=criterion,
    train_loader=train_loader)
```
- Learning rate: To control learning rate decrement
  - para: "lr_scheduler": {"type": "StepLR","args": {"step_size": 200,"gamma": 0.1}}
- Optimizer: 
  - para: {"type": "Adam"/"SGD","args": {"lr": 0.001,"weight_decay": 0,"amsgrad": true}}
- Some other parameters:
  - "epochs": 600,
  - "display_interval": 10,
  - "show_images_interval": 50,
  - "resume_checkpoint": "",     
  - "finetune_checkpoint": "",
  - "output_dir": "output",
  - "tensorboard": true,
  - "metrics": "hmean"
  
## Training phase:
```bash
trainer.train()
```
This trains the model, and saves latest and best checkpoint after each epochs.

