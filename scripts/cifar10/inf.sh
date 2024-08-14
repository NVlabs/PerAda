model_paths=( 
"resnet18_pretrain_fix_adapter_our_bs64lr0.01_mean_ncli20per8_lep10_a0.1_lmbda1.0"
"resnet18_pretrain_fix_adapter_our_bs64lr0.01_kd_cifar100kdbs2048lr0.001b500datafrac1max30_ncli20per8_lep10_a0.1_lmbda1.0"
)



folder="outputs/cifar10/"
dirichlet_alpha=0.1

for item in "${model_paths[@]}"
do
    echo $item
    # global test
    python infer.py --f $item --p $folder --dataset cifar10 --num_clients 20; 
    # local test
    python infer_per.py --f $item --p $folder --dirichlet_alpha ${dirichlet_alpha} --dataset cifar10 --num_clients 20; 
    # ood test
    python infer.py --f $item --p $folder --corrupt --dataset cifar10 --num_clients 20;    
done

