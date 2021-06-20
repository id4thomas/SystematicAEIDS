# Train Hyperparams
epoch=100
batch_size=512
lr=1e-4

# Model Args
#(5,32) Model
# num_layers=2
# max_hid_size=32

#(5,64) Model
num_layers=2
max_hid_size=64

#(7,64) Model
# num_layers=3
# max_hid_size=64

#Plot MCC, TPR
python plot_perf_nsl.py --num_layers $num_layers --max_hid_size $max_hid_size --batch_size $batch_size --epoch $epoch

#Plot Recon Dist
#Best Ldim: (32,5):4, (64,5):3, (64,7):9
python plot_dist_nsl_avg.py --l_dim 3 --num_layers $num_layers --max_hid_size $max_hid_size --batch_size $batch_size --epoch $epoch