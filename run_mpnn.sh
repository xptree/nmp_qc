#!/bin/bash

set -x
for target in mu alpha homo lumo gap r2 zpve U0 U H G Cv
do
    echo $target
    chk=checkpoint/QM9/$target
    mkdir -p $chk
    CUDA_VISIBLE_DEVICES=0 python main.py -l $target \
        --lr 0.00013 --lr-decay 0.5 \
        --resume $chk \
        --batch-size 20 --log-interval 200 \
        --epochs 200 |& tee update_$target.log
done
