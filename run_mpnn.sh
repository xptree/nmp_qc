#!/bin/bash

set -x
for target in mu alpha homo lumo gap r2 zpve U0 U H G Cv
do
    echo $target
    mkdir -p checkpoint/$target
    CUDA_VISIBLE_DEVICES=0 python main.py -l $target \
        --lr 0.00013 --lr-decay 0.5 \
        --resume checkpoint/$target \
        --batch-size 20 --log-interval 1000 |& tee $target.log
done
