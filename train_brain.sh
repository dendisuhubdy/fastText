#!/bin/bash

EXPNAME=`date "+%Y-%m-%d_%H-%M-%S_%N"`

mkdir -p exp/$EXPNAME
echo "Starting $EXPNAME"...

gdb --args fasttext skipgram \
    -input data/random_hist.txt \
    -output exp/$EXPNAME/model \
    -lr 0.05 \
    -dim 300 \
    -minCount 1 \
    -loss ns \
    -thread 1 \
    -epoch 2000 \
    -printEvery 10
