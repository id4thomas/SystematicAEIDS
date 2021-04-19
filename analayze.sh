#!/bin/bash
epoch=300
batch_size=8192
num_layers=2

# LDIM="1 2 5 8 10 27 32"
LDIM="1"
DIST="l2"
DATA="kyoto"
for i in $LDIM
do
    echo "Running loop seq "$i
    #balanced
    # python log_analysis.py --size 32 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --bal --dist $DIST --data $DATA
    python log_analysis.py --size 64 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --bal --dist $DIST --data $DATA
    # python log_analysis.py --size 128 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --bal --dist $DIST --data $DATA
    # python log_analysis.py --size 256 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --bal --dist $DIST --data $DATA

    #full
    # python log_analysis.py --size 32 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --dist $DIST --data $DATA
    # python log_analysis.py --size 64 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --dist $DIST --data $DATA
    # python log_analysis.py --size 128 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --dist $DIST --data $DATA
    # python log_analysis.py --size 256 --batch_size $batch_size --epoch $epoch --l_dim $i --num_layers $num_layers --dist $DIST --data $DATA
done
