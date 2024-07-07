python train.py --arch yolov8m --img-size 448 --num-epochs 100 --mosaic --close-mosaic 5 --model-ema --project brain-m-448 --dataset brain

python train.py --arch yolov8n --img-size 448 --num-epochs 100 --mosaic --close-mosaic 5 --model-ema --project catdog-n-448 --dataset catdog

python train.py --arch yolov8n --img-size 448 --num-epochs 100 --mosaic --close-mosaic 5 --model-ema --scratch --project catdog-n-448-scratch --dataset catdog

python train.py --arch yolov8s --img-size 448 --num-epochs 100 --mosaic --close-mosaic 5 --model-ema --project drive-s-448 --dataset drive
