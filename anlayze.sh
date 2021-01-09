#!/bin/bash
epoch=1000
batch_size=8192
num_layers=2

# LDIM="1 2 5 8 10 27 32"
LDIM="1"

for i in $LDIM
do
    echo "Running loop seq "$i
    python log_analysis.py --size 64 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers
    python log_analysis.py --size 128 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers
done
