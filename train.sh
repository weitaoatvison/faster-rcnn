#!/bin/bash
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET=GoogLeNet
#VGG16 GoogLeNet
NET_lc=${NET,,}
DATASET=kitti
# PRETRAINWEIGHT="./data/imagenet_models/VGG16.v2.caffemodel"
PRETRAINWEIGHT="./data/imagenet_models/GoogLeNet.caffemodel"

case $DATASET in
  kitti)
    TRAIN_IMDB="kitti_train"
    TEST_IMDB="kitti_test"
    PT_DIR="kitti"
    ITERS=150000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/solver.prototxt \
  --weights ${PRETRAINWEIGHT} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml
