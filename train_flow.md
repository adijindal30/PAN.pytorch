# Train.py

## Initialization:

(a) Loading the data:
```bash
train_loader = get_dataloader(parameters)
```
- The parameters are train_path, val_path, val_ratio, input_size etc. Can be changes into config.json\
- This calls data_loader/_init_.py, which further calls data_loader/dataset.py for Imagedatset which augment the data as its resize,crop or horizontally flip 
or can add noise for more generic model.

(b) Setting up PAN loss:
```bash
criterion = get_loss(config).cuda()
```
- This calls to model/loss.py where PANLoss initialize with hyperparameters as :\
alpha=0.5, beta=0.25, delta_agg=0.5, delta_dis=3, ohem_ratio=3, reduction='mean'
- This also setups up training masks, kernel coordinates, text coordinates which use to calculate loss for our text detection model


(c) Setting up the Deep-Learning model:
```bash
model = get_model(config)
```

