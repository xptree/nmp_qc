#!/bin/bash

set -x
for target in mu lumo H
do
    echo $target
    chk=checkpoint/QM9/$target
    mkdir -p $chk
    CUDA_VISIBLE_DEVICES=1 python main.py -l $target \
        --lr 0.00013 --lr-decay 0.5 \
        --resume $chk \
        --aichemy-path ./data/aichemy/ \
        --batch-size 20 --log-interval 200 \
        --epochs 0 |& tee test_$target.log
done
