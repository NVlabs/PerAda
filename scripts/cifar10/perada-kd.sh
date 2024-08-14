batch_size=64
num_epochs=10 # number of local epochs
clients_per_round=8
num_clients=20
dirichlet_alpha=0.1
num_rounds=200
lr=0.01
lr_decay=1
decay_lr_every=50
lmbda=1.0
dataset="cifar10"


kd_dataset="cifar100"
kd_lr=0.001 # server learning rate for KD
kd_max_round=30
kd_batch_size=2048
kd_eval_batches_freq=1
total_n_server_pseudo_batches=500  
early_stopping_server_batches=1000
kd_data_fraction=1

outf="outputs/cifar10/"

pfl_algo='our'
aggregation='kd'
net='resnet18_pretrain_fix_adapter'

options="-dataset ${dataset} --pfl_algo ${pfl_algo}  --kd_data_fraction ${kd_data_fraction} --net ${net} --personalized --log_online --project pfl-cifar10  --lmbda ${lmbda} --batch_size ${batch_size} --num_clients ${num_clients} --dirichlet_alpha ${dirichlet_alpha} --clients-per-round ${clients_per_round} --num-rounds ${num_rounds} -lr ${lr} --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} --num_epochs ${num_epochs} --aggregation ${aggregation} --kd_dataset ${kd_dataset} --kd_lr ${kd_lr} --total_n_server_pseudo_batches ${total_n_server_pseudo_batches} --early_stopping_server_batches ${early_stopping_server_batches} --kd_eval_batches_freq ${kd_eval_batches_freq} --kd_max_round ${kd_max_round} --kd_batch_size ${kd_batch_size}"
optfname="${outf}${net}_${pfl_algo}_bs${batch_size}lr${lr}_${aggregation}_${kd_dataset}kdbs${kd_batch_size}lr${kd_lr}b${total_n_server_pseudo_batches}datafrac${kd_data_fraction}max${kd_max_round}_ncli${num_clients}per${clients_per_round}_lep${num_epochs}_a${dirichlet_alpha}_lmbda${lmbda}"
python run.py $options --output_summary_file $optfname





