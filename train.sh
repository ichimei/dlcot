CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone resnet --lr 0.007 --workers 2 --epochs 50 --checkname deeplab-resnet --eval-interval 1 --dataset pascal
