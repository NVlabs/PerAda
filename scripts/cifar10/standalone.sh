batch_size=64
num_epochs=10
clients_per_round=8
num_clients=20
dirichlet_alpha=0.1
num_rounds=200
lr=0.01
lr_decay=1
decay_lr_every=50
lmbda=1.0
dataset="cifar10"
outf="outputs/cifar10/"


net='resnet18_pretrain'
pfl_algo='standalone' 

options="-dataset ${dataset} -model ${model} --net ${net} --pfl_algo ${pfl_algo} --personalized --log_online --project pfl-cifar10 --lmbda ${lmbda} --batch_size ${batch_size} --num_clients ${num_clients} --dirichlet_alpha ${dirichlet_alpha} --clients-per-round ${clients_per_round} --num-rounds ${num_rounds} -lr ${lr} --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} --num_epochs ${num_epochs}"
optfname="${outf}${net}_${pfl_algo}_bs${batch_size}lr${lr}_ncli${num_clients}per${clients_per_round}_lep${num_epochs}_a${dirichlet_alpha}_lmbda${lmbda}"
python run.py $options --output_summary_file $optfname




