# <div align="center">easy-YOLOv8</div>

### Description

This is a repository for implementation of YOLOv8 for easy customization and understanding underlying techniques in it, which is refered to ultralytics' YOLOv8 (https://github.com/ultralytics/ultralytics).   


### User Command 

You can train your own YOLOv8 model with command like below. As for <DATASET> you can refer sample file in cfg/*.yaml, and make <DATASET NAME>.yaml file following your dataset. Since cfg/*.json file that is required to compute mAP scores is built automatically via dataloader, you do not have to worry about it. 

 - **Pretrained Model Weights Download**

	- [YOLOv8-n/s/m/l/x](https://drive.google.com/drive/folders/15ZSlGSijAzqIyV5PjmMKIOhKhXmemFOy?usp=sharing)


```python

# Training
python train.py --arch yolov8n --img-size 640 --num-epochs 200 --mosaic --cos-lr --model-ema --project <YOUR PROJECT> --dataset <YOUR DATASET>

# Evaluation
python val.py --project <YOUR PROJECT>

# Inference in images
python test.py --project <YOUR PROJECT> --test-dir <IMAGE DIRECTORY>

# Inference in video
python infer.py --project <YOUR PROJECT> --vid_path <VIDEO PATH>
```

---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  