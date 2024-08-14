batch_size=256
num_epochs=1
clients_per_round=8
num_clients=20
dirichlet_alpha=0.3
num_rounds=30
eval_every=3
lr=0.01
lr_decay=1
decay_lr_every=50
lmbda=1.0
dataset="chexpert"
outf="outputs/chexpert/"


kd_dataset="cifar10"
kd_lr=0.00001
kd_max_round=5
kd_batch_size=128
kd_eval_batches_freq=10
total_n_server_pseudo_batches=50
early_stopping_server_batches=1000

pfl_algo='our'
aggregation='kd'
net='resnet18_pretrain_fix_adapter'

options="-dataset ${dataset} -model ${model} --eval-every ${eval_every} --pfl_algo ${pfl_algo}  --net ${net} --personalized --log_online --project pfl-chexpert  --lmbda ${lmbda} --batch_size ${batch_size} --num_clients ${num_clients} --dirichlet_alpha ${dirichlet_alpha} --clients-per-round ${clients_per_round} --num-rounds ${num_rounds} -lr ${lr} --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} --num_epochs ${num_epochs} --aggregation ${aggregation} --kd_dataset ${kd_dataset} --kd_lr ${kd_lr}  --total_n_server_pseudo_batches ${total_n_server_pseudo_batches} --early_stopping_server_batches ${early_stopping_server_batches} --kd_eval_batches_freq ${kd_eval_batches_freq} --kd_max_round ${kd_max_round} --kd_batch_size ${kd_batch_size}"
optfname="${outf}${net}_${pfl_algo}_bs${batch_size}lr${lr}_${aggregation}_${kd_dataset}kdbs${kd_batch_size}lr${kd_lr}b${total_n_server_pseudo_batches}max${kd_max_round}_ncli${num_clients}per${clients_per_round}_lep${num_epochs}_a${dirichlet_alpha}_lmbda${lmbda}"
python run.py $options --output_summary_file $optfname



