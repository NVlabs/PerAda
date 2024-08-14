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


pfl_algo='fedavg'
net='resnet18_pretrain'
aggregation='mean'

options="-dataset ${dataset} --eval-every ${eval_every} --pfl_algo ${pfl_algo}  --net ${net} --personalized --log_online --project pfl-chexpert  --lmbda ${lmbda} --batch_size ${batch_size} --num_clients ${num_clients} --dirichlet_alpha ${dirichlet_alpha} --clients-per-round ${clients_per_round} --num-rounds ${num_rounds} -lr ${lr} --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} --num_epochs ${num_epochs} --aggregation ${aggregation}"
optfname="${outf}${net}_${pfl_algo}_bs${batch_size}lr${lr}_${aggregation}_ncli${num_clients}per${clients_per_round}_lep${num_epochs}_a${dirichlet_alpha}_lmbda${lmbda}"
python run.py $options --output_summary_file $optfname
