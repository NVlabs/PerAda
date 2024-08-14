model_paths=( 
"resnet18_pretrain_fix_adapter_our_bs256lr0.01_mean_ncli20per8_lep1_a0.3_lmbda1.0"
"resnet18_pretrain_fix_adapter_our_bs256lr0.01_kd_cifar10kdbs128lr0.00001b50max5_ncli20per8_lep1_a0.3_lmbda1.0"
)


folder="outputs/chexpert"

dirichlet_alpha=0.3

for item in "${model_paths[@]}"
do
    echo $item
    # local test
    python infer_per.py --f $item --p $folder --dataset chexpert --num_clients 20 --model resnet18 --dirichlet_alpha ${dirichlet_alpha} ; 
    # global test
    python infer.py --f $item --p $folder --dataset chexpert --num_clients 20 --model resnet18;  
done
