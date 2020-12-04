wandb login $WANDB_API_KEY

python ./manage.py \
    --gpu 0 \
    --client_num_in_total 2 \
    --client_num_per_round 2 \
    --batch_size 4 \
    --dataset fed_cifar100 \
    --data_dir C:/Users/Johan/Workspace/FedML/data/fed_cifar100 \
    --model resnet18_gn \
    --partition_method hetero \
    --comm_round 1 \
    --epochs 1 \
    --lr 0.03 \
    --client_optimizer adam \
    --ci 0
