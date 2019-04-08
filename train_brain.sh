#!/bin/bash

EXPNAME=`date "+%Y-%m-%d_%H-%M-%S_%N"`

mkdir -p exp/$EXPNAME
echo "Starting $EXPNAME"...

fasttext skipgram \
    -input /home/ubuntu/data/customer_interest_matrix.txt \
    -output exp/$EXPNAME/model \
    -lr 0.05 \
    -dim 300 \
    -minCount 1 \
    -loss ns \
    -thread 8 \
    -epoch 2000 2>&1 | tee -i exp/$EXPNAME/log
