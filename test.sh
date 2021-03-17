SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
PYTHONPATH=$(dirname "$SCRIPTPATH")
python test.py --in-path input.jpg --out-path output.png --model my.pth --gpu 0
