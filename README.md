# <div align="center">easy-YOLOv8</div>

### Description

This is a repository for implementation of YOLOv8 for easy customization and understanding underlying techniques in it, which is refered to ultralytics' YOLOv8 (https://github.com/ultralytics/ultralytics).   


### User Command 

You can train your own YOLOv8 model with command like below. As for <DATASET> you can refer sample file in cfg/*.yaml, and make <DATASET NAME>.yaml file following your dataset. Since cfg/*.json file that is required to compute mAP scores is built automatically via dataloader, you do not have to worry about it. 

 - **Pretrained Model Weights Download**

	- [YOLOv8-n/s/m/l/x](https://drive.google.com/drive/folders/15ZSlGSijAzqIyV5PjmMKIOhKhXmemFOy?usp=sharing)


| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | mAP<br><sup>(@0.5:0.95) | Params<br><sup>(M) | FLOPs<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| YOLOv8n | COCO | train2017 | val2017 | 640 | 37.3 | 3.2 | 8.7 |
| YOLOv8s | COCO | train2017 | val2017 | 640 | 44.9 | 11.2 | 28.6 |
| YOLOv8m | COCO | train2017 | val2017 | 640 | 50.2 | 25.9 | 78.9 |
| YOLOv8l | COCO | train2017 | val2017 | 640 | 52.9 | 43.7 | 165.2 |
| YOLOv8x | COCO | train2017 | val2017 | 640 | 53.9 | 68.2 | 257.8 |


```python

# Training
python train.py --arch yolov8n --img-size 640 --num-epochs 200 --mosaic --close-mosaic 5 --model-ema --project <YOUR PROJECT> --dataset <YOUR DATASET>

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