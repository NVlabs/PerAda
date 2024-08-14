model_paths=( 
"resnet18_pretrain_fix_adapter_our_bs128lr0.05_mean_ncli4per4_lep1_lmbda0.1"
"resnet18_pretrain_fix_adapter_our_bs128lr0.05_kd_cifar10kdbs256lr0.0001b100max30_ncli4per4_lep1_lmbda0.1"
)


folder="outputs/oh"

for item in "${model_paths[@]}"
do
    echo $item
    # global test
    python infer.py --f $item --p $folder --dataset oh --num_clients 4 --model resnet18; 
    # lcoal test
    python infer_per.py --f $item --p $folder --dataset oh --num_clients 4 --model resnet18; 
done

