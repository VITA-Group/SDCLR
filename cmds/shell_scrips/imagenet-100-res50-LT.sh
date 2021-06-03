# get opts
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -e|--epochs) pretrain_epochs="$2"; shift; shift ;;
    -s|--split) pretrain_split="$2"; shift; shift ;;
    -p|--port) port="$2"; shift; shift ;;
    -w|--workers) workers="$2"; shift; shift ;;
    -g|--GPU_NUM) GPU_NUM=("$2"); shift; shift ;;
    --data) data=("$2"); shift; shift ;;
    --seed) seed=("$2"); shift; shift ;;
    --temp) temp=("$2"); shift; shift ;;
    --pretrain_lr) pretrain_lr=("$2"); shift; shift ;;
    --prune) prune=("$2"); shift; shift ;;
    --prune_percent) prune_percent=("$2"); shift; shift ;;
    --prune_dual_bn) prune_dual_bn=("$2"); shift; shift ;;
    --pretrain_batch_size) pretrain_batch_size=("$2"); shift; shift ;;
    --only_pretraining) only_pretraining=("$2"); shift; shift ;;
    --only_finetuning) only_finetuning=("$2"); shift; shift ;;
    --test_only) test_only=("$2"); shift; shift ;;
    --only_few_shot) only_few_shot=("$2"); shift; shift ;;
    --few_shot_split) few_shot_split=("$2"); shift; shift ;;
    --linear_eval_seed) linear_eval_seed=("$2"); shift; shift ;;
    *) echo "${1} is not found"; exit 125;
esac
done


port=${port:-4833}
pretrain_epochs=${pretrain_epochs:-500}
pretrain_split=${pretrain_split:-imageNet_100_LT_train}
workers=${workers:-10}
pretrain_batch_size=${pretrain_batch_size:-256}
linear_eval_seed=${linear_eval_seed:-1}
prune=${prune:-False}
prune_percent=${prune_percent:-0.3}
temp=${temp:-0.2}
pretrain_lr=${pretrain_lr:-0.5}
prune_dual_bn=${prune_dual_bn:-False}
only_pretraining=${only_pretraining:-False}
only_finetuning=${only_finetuning:-False}
test_only=${test_only:-False}
only_few_shot=${only_few_shot:-False}
few_shot_split=${few_shot_split:-imageNet_100_sub_balance_train_0.01}
data=${data:-placeholder}
seed=${seed:-1}
save_dir=checkpoints_imagenet

pretrain_name=${pretrain_split}_res50_scheduling_sgd_lr${pretrain_lr}_temp${temp}_epoch${pretrain_epochs}_batch${pretrain_batch_size}

if [[ ${prune} == "True" ]]
then
  pretrain_name="${pretrain_name}_pruneP${prune_percent}"

  if [[ ${prune_dual_bn} == "True" ]]
  then
    pretrain_name="${pretrain_name}DualBN"
  fi
fi

cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} train_simCLR.py \
${pretrain_name} --epochs ${pretrain_epochs} \
--batch_size ${pretrain_batch_size} --output_ch 128 --lr ${pretrain_lr} --temperature ${temp} --model res50 \
--dataset imagenet-100 --imagenetCustomSplit ${pretrain_split} --save-dir ${save_dir} --optimizer sgd \
--num_workers ${workers} --seed ${seed} --data ${data}"

if [[ ${prune} == "True" ]]
then
  cmd="${cmd} --prune --prune_percent ${prune_percent}"

  if [[ ${prune_dual_bn} == "True" ]]
  then
    cmd="${cmd} --prune_dual_bn"
  fi

fi

tuneLr=30
cmd_full="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} \
train_imagenet.py ${pretrain_name}__lr${tuneLr}_wd0_epoch30_b512_d10d20_s${linear_eval_seed} \
--decreasing_lr 10,20 --weight-decay 0 --epochs 30 --lr ${tuneLr} --batch-size 512 \
--model res50 --fullset --save-dir checkpoints_imagenet_tune --dataset imagenet-100 \
--checkpoint checkpoints_imagenet/${pretrain_name}/model_${pretrain_epochs}.pt \
--cvt_state_dict --world_size ${GPU_NUM} --port ${port} --num_workers ${workers} --test_freq 2 \
--seed ${linear_eval_seed} --data ${data}"

if [[ ${prune_dual_bn} == "True" ]]
then
  cmd_full="${cmd_full} --bnNameCnt 0"
fi

if [[ ${test_only} == "True" ]]
then
  cmd_full="${cmd_full} --test_only"
fi

tuneLr=30
cmd_few_shot="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} \
  train_imagenet.py ${pretrain_name}__${few_shot_split}_lr${tuneLr}_fix2layer4_wd0_epoch30_b512_d10d20 \
  --decreasing_lr 40,60 --weight-decay 0 --epochs 100 --lr ${tuneLr} --batch-size 512 \
  --model res50 --save-dir checkpoints_imagenet_tune --dataset imagenet-100 --customSplit ${few_shot_split} \
  --checkpoint checkpoints_imagenet/${pretrain_name}/model_${pretrain_epochs}.pt \
  --cvt_state_dict --world_size ${GPU_NUM} --port ${port} --num_workers ${workers} --test_freq 10 --data ${data}"

if [[ ${prune_dual_bn} == "True" ]]
then
  cmd_few_shot="${cmd_few_shot} --bnNameCnt 0"
fi

if [[ ${test_only} == "True" ]]
then
  cmd_few_shot="${cmd_few_shot} --test_only"
fi

mkdir -p ${save_dir}/${pretrain_name}

if [[ ${only_few_shot} == "False" ]]
then
  if [[ ${only_finetuning} == "False" ]]
  then
    echo ${cmd} >> ${save_dir}/${pretrain_name}/bash_log.txt
    echo ${cmd}
    ${cmd}
  fi

  if [[ ${only_pretraining} == "False" ]]
  then
    echo ${cmd_full} >> ${save_dir}/${pretrain_name}/bash_log.txt
    echo ${cmd_full}
    ${cmd_full}

    echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
    echo ${cmd_few_shot}
    ${cmd_few_shot}
  fi
else
  echo ${cmd_few_shot} >> ${save_dir}/${pretrain_name}/bash_log.txt
  echo ${cmd_few_shot}
  ${cmd_few_shot}
fi
