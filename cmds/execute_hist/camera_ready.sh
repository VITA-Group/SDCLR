# cifar 10
CUDA_VISIBLE_DEVICES=0 ./cmds/shell_scrips/cifar-10-LT.sh -g 1 -p 4866 -w 8 --split split1_D_b
CUDA_VISIBLE_DEVICES=0 ./cmds/shell_scrips/cifar-10-LT.sh -g 1 -p 4866 -w 8 --split split1_D_i
CUDA_VISIBLE_DEVICES=0 ./cmds/shell_scrips/cifar-10-LT.sh -g 1 -p 4867 -w 8 --split split1_D_i --prune True --prune_percent 0.9 --prune_dual_bn True

# cifar 100
CUDA_VISIBLE_DEVICES=0 ./cmds/shell_scrips/cifar-100-LT.sh -g 1 -p 4867 -w 8 --split cifar100_split1_D_b
CUDA_VISIBLE_DEVICES=0 ./cmds/shell_scrips/cifar-100-LT.sh -g 1 -p 4867 -w 8 --split cifar100_split1_D_i
CUDA_VISIBLE_DEVICES=0 ./cmds/shell_scrips/cifar-100-LT.sh -g 1 -p 4867 -w 8 --split cifar100_split1_D_i --prune True --prune_percent 0.9 --prune_dual_bn True

./cmds/shell_scrips/imagenet-100-res50-LT.sh -g 2 -p 4867 -w 10 --prune True --prune_percent 0.3 --temp 0.3 --prune_dual_bn True
