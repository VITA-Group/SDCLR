# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--epochs) pretrain_epochs="$2"; shift; shift ;;
    -s|--split) pretrain_split="$2"; shift; shift ;;
    -p|--port) port="$2"; shift; shift ;;
    -w|--workers) workers="$2"; shift; shift ;;
    -g|--GPU_NUM) GPU_NUM="$2"; shift; shift ;;
    --data) data="$2"; shift; shift ;;
    --pretrain_batch_size) pretrain_batch_size="$2"; shift; shift ;;
    --only_finetuning) only_finetuning="$2"; shift; shift ;;
    --test_only) test_only="$2"; shift; shift ;;
    *) echo "${1} is not found"; exit 125;
esac
done


port=${port:-4833}
data=${data:-placeholder}
pretrain_epochs=${pretrain_epochs:-500}
pretrain_split=${pretrain_split:-imageNet_LT_exp_train}
workers=${workers:-10}
pretrain_batch_size=${pretrain_batch_size:-256}
prune_percent=${prune_percent:-0.7}
only_finetuning=${only_finetuning:-False}
test_only=${test_only:-False}
save_dir=checkpoints_imagenet

pretrain_name=${pretrain_split}_res50_scheduling_sgd_lr0.5_temp0.2_epoch${pretrain_epochs}_batch${pretrain_batch_size}

launch_cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port}"

cmd="${launch_cmd} train_simCLR.py \
${pretrain_name} --epochs ${pretrain_epochs} \
--batch_size ${pretrain_batch_size} --output_ch 128 --lr 0.5 --temperature 0.2 --model res50 \
--dataset imagenet-LT --imagenetCustomSplit ${pretrain_split} --save-dir ${save_dir} --optimizer sgd \
--num_workers ${workers} --data ${data}"

tuneLr=30
cmd_full="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
train_imagenet.py ${pretrain_name}__lr${tuneLr}_wd0_epoch30_b512_d10d20 \
--decreasing_lr 10,20 --weight-decay 0 --epochs 30 --lr ${tuneLr} --batch-size 512 \
--model res50 --fullset --save-dir checkpoints_imagenet_tune --dataset imagenet \
--checkpoint checkpoints_imagenet/${pretrain_name}/model_${pretrain_epochs}.pt \
--cvt_state_dict --world_size ${GPU_NUM} --port ${port} --num_workers ${workers} --test_freq 2 --data ${data}"

if [[ ${test_only} == "True" ]]
then
  cmd_full="${cmd_full} --test_only"
fi

tuneLr=30
train_split1=imageNet_sub_balance_train_0.01
cmd_few_shot="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
  train_imagenet.py ${pretrain_name}__${train_split1}_lr${tuneLr}_fix2layer4_wd0_epoch30_b512_d10d20 \
  --decreasing_lr 40,60 --weight-decay 0 --epochs 100 --lr ${tuneLr} --batch-size 512 \
  --model res50 --save-dir checkpoints_imagenet_tune --dataset imagenet --customSplit ${train_split1} \
  --checkpoint checkpoints_imagenet/${pretrain_name}/model_${pretrain_epochs}.pt \
  --cvt_state_dict --world_size ${GPU_NUM} --port ${port} --num_workers ${workers} --test_freq 10 --data ${data}"

if [[ ${test_only} == "True" ]]
then
  cmd_few_shot="${cmd_few_shot} --test_only"
fi


mkdir -p ${save_dir}/${pretrain_name}

if [[ ${only_finetuning} == "False" ]]
then
  echo ${cmd} >> ${save_dir}/${pretrain_name}/bash_log.txt
  echo ${cmd}
  ${cmd}
fi

echo ${cmd_full} >> ${save_dir}/${pretrain_name}/bash_log.txt
echo ${cmd_full}
${cmd_full}

echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
echo ${cmd_few_shot}
${cmd_few_shot}
