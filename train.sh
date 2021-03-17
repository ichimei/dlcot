SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PYTHONPATH=$(dirname "$SCRIPTPATH")
python train.py --backbone resnet --lr 0.007 --workers 2 --epochs 50 --eval-interval 1 --dataset pascal --gpu-ids 0,1
