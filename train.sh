#!/bin/bash
epoch=1000
lr=1e-4
batch_size=8192
num_layers=3
SEEDS="10 20 30 40 50"
# SEEDS="10"
# SEEDS="20 30 40 50"
LDIM="1 2 5 8 10 27 32"
# LDIM="1"
for seed in $SEEDS
do
    for i in $LDIM
    do
        echo "Running loop seq "$i
        python train_model.py --batch_size $batch_size --epoch $epoch --l_dim $i --lr $lr --seed $seed --num_layers $num_layers
    done
done

# python train_model.py --seed 10 --epoch 10 --batch_size 8192 --l_dim 32 --num_layers 4