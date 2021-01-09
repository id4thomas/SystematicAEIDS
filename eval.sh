#!/bin/bash
epoch=500
lr=1e-4
batch_size=8192
num_layers=2
SEEDS="10 20 30 40 50"
SEEDS="10"
SEEDS="10 30 50"
LDIM="1 2 5 8 10 27 32"
# LDIM="1"
for seed in $SEEDS
do
    for i in $LDIM
    do
        echo "Running loop seq "$i
        # python eval_model.py --size 64 --batch_size $batch_size --epoch $epoch --l_dim $i --lr $lr --seed $seed --num_layers $num_layers
        python eval_model.py --size 128 --batch_size $batch_size --epoch $epoch --l_dim $i --lr $lr --seed $seed --num_layers $num_layers
    done
done

LDIM="1"
for seed in $SEEDS
do
    for i in $LDIM
    do
        echo "Running loop seq "$i
        # python log_analysis.py --size 64 --batch_size $batch_size --epoch $epoch --l_dim $i --seed $seed --num_layers $num_layers
        python log_analysis.py --size 128 --batch_size $batch_size --epoch $epoch --l_dim $i  --seed $seed --num_layers $num_layers
    done
done