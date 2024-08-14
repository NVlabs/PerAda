batch_size=128
num_epochs=1 # number of local epochs
clients_per_round=4
num_clients=4
num_rounds=100
lr=0.05
lr_decay=1
decay_lr_every=50
lmbda=0.1
dataset="oh"

outf="outputs/oh/"

pfl_algo='standalone'
net='resnet18_pretrain'


options="-dataset ${dataset}  --pfl_algo ${pfl_algo}  --net ${net} --personalized --log_online --project pfl-office --lmbda ${lmbda} --batch_size ${batch_size} --num_clients ${num_clients} --clients-per-round ${clients_per_round} --num-rounds ${num_rounds} -lr ${lr} --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} --num_epochs ${num_epochs}"
optfname="${outf}${net}_${pfl_algo}_bs${batch_size}lr${lr}_ncli${num_clients}per${clients_per_round}_lep${num_epochs}_lmbda${lmbda}"
python run.py $options --output_summary_file $optfname
