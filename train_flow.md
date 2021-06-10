# Train.py

## Initialization:

(a) Loading the data:
```bash
train_loader = get_dataloader(parameters)
```
- The parameters are train_path, val_path, val_ratio, input_size etc. Can be changes into config.json\
- This calls data_loader/_init_.py, which further calls data_loader/dataset.py for Imagedatset which augment the data as its resize,crop or horizontally flip 
or can add noise for more generic model.
- 
