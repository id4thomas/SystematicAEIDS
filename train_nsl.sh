# Train Hyperparams
epoch=100
batch_size=512
lr=1e-4

# Model Args
#(5,32) Model
# num_layers=2
# max_hid_size=32
# LDIM="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

#(5,64) Model
# num_layers=2
# max_hid_size=64
# LDIM="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"

#(7,64) Model
num_layers=3
max_hid_size=64
LDIM="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

#20 test Runs
SEEDS="10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200"

for seed in $SEEDS
do
    for i in $LDIM
    do
        echo "Running loop seq "$i
        python train_nsl.py --num_layers $num_layers --max_hid_size $max_hid_size --l_dim $i --batch_size $batch_size --epoch $epoch --lr $lr --seed $seed 
        python eval_nsl.py --num_layers $num_layers --max_hid_size $max_hid_size --l_dim $i --batch_size $batch_size --epoch $epoch --seed $seed 
    done
done

#TPR per Attack Category
python eval_nsl_cat_tpr.py --num_layers $num_layers --max_hid_size $max_hid_size --batch_size $batch_size --epoch $epoch --num_runs 20