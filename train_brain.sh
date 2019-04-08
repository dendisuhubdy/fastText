#!/bin/bash

EXPNAME=`date "+%Y-%m-%d_%H-%M-%S_%N"`

mkdir -p exp/$EXPNAME
echo "Starting $EXPNAME"...

fasttext skipgram \
    -input data/bidder_.txt \
    -output exp/$EXPNAME/model \
    -lr 0.05 \
    -dim 100 \
    -minCount 1 \
    -loss ns \
    -thread 8 \
    -epoch 10000 \
    -printEvery 10 2>&1 | tee -i exp/$EXPNAME/log
