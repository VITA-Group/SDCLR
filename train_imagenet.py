from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from utils import setup_seed

from models.resnet import resnet18, resnet10, resnet50
import time
from utils import AverageMeter, logger
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from data.LT_Dataset import LT_Dataset
import torch.distributed as dist
from pdb import set_trace
from utils import accuracy, getImagenetRoot
import re
from utils import getStatisticsFromTxt
from collections import OrderedDict
from train_simCLR import cosine_annealing

parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
parser.add_argument('experiment', type=str, help='exp name')
parser.add_argument('--model', default='res50', type=str, help='model name')
parser.add_argument('--data', default='', type=str, help='path to data')
parser.add_argument('--dataset', default='imagenet', type=str, help='imagenet, imagenet-100 ')
parser.add_argument('--save-dir', default='checkpoints_imagent_cls', type=str, help='path to save checkpoint')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')
parser.add_argument('--resume', action='store_true',
                    help='if resume training')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='the start epoch number')
parser.add_argument('--log-interval', default=50, type=int,
                    help='display interval')
parser.add_argument('--decreasing_lr', default='3,6,9', help='decreasing strategy')
parser.add_argument('--cvt_state_dict', action='store_true', help='use for ss model')
parser.add_argument('--fullset', action='store_true', help='if use the full set')
parser.add_argument('--customSplit', type=str, default='', help='custom split for training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--num_workers', type=int, default=10, help='num workers')
parser.add_argument('--bnNameCnt', default=-1, type=int,
                    help='bnNameCnt, for pretraining with prune_dual_bn')
parser.add_argument('--test_only', action='store_true')


# unfreeze
parser.add_argument('--unfreeze', action='store_true', help='if unfreeze the model')
# distributed training
parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')
parser.add_argument('--world_size', default=1, type=int, help='world size')
parser.add_argument('--port', default=4850, type=int, help='sync port')
parser.add_argument('--test_freq', default=1, help="test freq", type=int)

args = parser.parse_args()

world_size = args.world_size
print("employ {} gpus".format(world_size))

assert args.batch_size % world_size == 0
batch_size = args.batch_size // world_size

# settings
model_dir = os.path.join(args.save_dir, args.experiment)
if not os.path.exists(model_dir) and args.local_rank == 0:
  os.makedirs(model_dir)
log = logger(os.path.join(model_dir), local_rank=args.local_rank)
log.info(str(args))
use_cuda = torch.cuda.is_available()
setup_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# distributed
url = 'tcp://127.0.0.1:{}'.format(args.port)
dist.init_process_group(backend="nccl", init_method=url, rank=args.local_rank, world_size=world_size)
torch.cuda.set_device(args.local_rank)

if args.dataset == 'imagenet' or args.dataset == 'imagenet-100':
  # setup data loader
  transform_train = transforms.Compose([
              transforms.RandomResizedCrop(224),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
          ])
  transform_test = transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
          ])
else:
  assert False


root = getImagenetRoot(args.data)

if args.dataset == 'imagenet':
  txt_train = "split/ImageNet_LT/ImageNet_LT_train.txt"
  txt_val = "split/ImageNet_LT/ImageNet_LT_val.txt"
  txt_test = "split/ImageNet_LT/ImageNet_LT_test.txt"
elif args.dataset == 'imagenet-100':
  txt_train = "split/imagenet-100/ImageNet_100_train.txt"
  txt_val = "split/imagenet-100/ImageNet_100_val.txt"
  txt_test = "split/imagenet-100/ImageNet_100_test.txt"
else:
  assert False

assert not (args.fullset and args.customSplit != '')

if args.fullset:
  if args.dataset == 'imagenet':
    txt_train = "split/ImageNet_LT/ImageNet_train.txt"
  elif args.dataset == 'imagenet-100':
    txt_train = "split/imagenet-100/ImageNet_100_train.txt"
  else:
    assert False

if args.customSplit != '':
  if args.dataset == 'imagenet':
    txt_train = "split/ImageNet_LT/{}.txt".format(args.customSplit)
  elif args.dataset == 'imagenet-100':
    txt_train = "split/imagenet-100/{}.txt".format(args.customSplit)
  else:
    assert False

if args.data != '':
  root = args.data

train_datasets = LT_Dataset(root=root, txt=txt_train, transform=transform_train)
val_datasets = LT_Dataset(root=root, txt=txt_val, transform=transform_test)
test_datasets = LT_Dataset(root=root, txt=txt_test, transform=transform_test)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=True)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_datasets, shuffle=False)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_datasets, shuffle=False)

train_loader = torch.utils.data.DataLoader(train_datasets, num_workers=args.num_workers, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_datasets, num_workers=4, batch_size=batch_size, sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(test_datasets, num_workers=4, batch_size=batch_size, sampler=test_sampler)

if args.dataset == 'imagenet':
  num_class = 1000
elif args.dataset == 'imagenet-100':
  num_class = 100
else:
  assert False

if args.local_rank == 0:
  class_stat = [0 for _ in range(num_class)]
  for target in train_datasets.labels:
      class_stat[target] += 1
  print("class distribution in training set is {}".format(class_stat))


def train(args, model, device, train_loader, optimizer, epoch, log, world_size, scheduler):
  model.train()

  for name, m in model.named_modules():
      if not ("fc" in name):
          m.eval()

  dataTimeAve = AverageMeter()
  totalTimeAve = AverageMeter()
  end = time.time()

  for batch_idx, (data, target, _) in enumerate(train_loader):

    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
    dataTime = time.time() - end
    dataTimeAve.update(dataTime)

    optimizer.zero_grad()

    logits = model(data)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()

    totalTime = time.time() - end
    totalTimeAve.update(totalTime)
    end = time.time()
    # print progress
    if batch_idx % args.log_interval == 0:
      log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f} ({:.3f})\tTotal time: {:.3f} ({:.3f})'.format(
        epoch, (batch_idx * train_loader.batch_size + len(data)) * world_size, len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.val, dataTimeAve.avg, totalTimeAve.val, totalTimeAve.avg))


def eval_test(model, device, loader, log, world_size, prefix='test', num_class=1000):
  model.eval()
  test_loss = 0
  correct = 0
  whole = 0

  top1_avg = AverageMeter()
  top5_avg = AverageMeter()
  model.eval()

  perClassAccRight = [0 for i in range(num_class)]
  perClassAccWhole = [0 for i in range(num_class)]

  with torch.no_grad():
    for data, target, _ in loader:
      data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
      output = model(data)

      output_list = [torch.zeros_like(output) for _ in range(world_size)]
      target_list = [torch.zeros_like(target) for _ in range(world_size)]
      torch.distributed.all_gather(output_list, output)
      torch.distributed.all_gather(target_list, target)
      output = torch.cat(output_list, dim=0)
      target = torch.cat(target_list, dim=0)

      pred = output.max(1)[1].long()
      for cntClass in torch.unique(target):
        perClassAccRight[cntClass] += pred[target == cntClass].eq(
          target[target == cntClass].view_as(pred[target == cntClass])).sum().item()
        perClassAccWhole[cntClass] += len(target[target == cntClass])

      test_loss += F.cross_entropy(output, target, size_average=False).item()
      # pred = output.max(1, keepdim=True)[1]
      # correct += pred.eq(target.view_as(pred)).sum().item()
      # whole += len(target)
      top1, top5 = accuracy(output, target, topk=(1,5))
      top1_avg.update(top1, data.shape[0])
      top5_avg.update(top5, data.shape[0])

  classWiseAcc = np.array(perClassAccRight) / np.array(perClassAccWhole)
  accPerClassStr = ""
  for i in range(num_class):
    accPerClassStr += "{:.04} ".format(classWiseAcc[i])
  log.info('acc per class is {}'.format(accPerClassStr))
  test_loss /= len(loader.dataset)
  log.info('{}: Average loss: {:.4f}, Accuracy: top1 ({:.2f}%) top5 ({:.2f}%)'.format(prefix,
    test_loss, top1_avg.avg, top5_avg.avg))
  return test_loss, top1_avg.avg, top5_avg.avg


def fix_model(model, log):
    # fix every layer except fc
    # fix previous four layers
    for name, param in model.named_parameters():
      log.info(name)
      if not ("fc" in name):
        log.info("fix {}".format(name))
        param.requires_grad = False


def main():
  # init model, ResNet18() can be also used here for training
  # do not use imagenet mode for imagenet32
  imagenet = args.dataset == 'imagenet' or args.dataset == 'imagenet-100'
  if args.model == 'res18':
    model = resnet18(num_classes=num_class, imagenet=imagenet).cuda()
  elif args.model == 'res10':
    model = resnet10(num_classes=num_class, imagenet=imagenet).cuda()
  elif args.model == 'res50':
    model = resnet50(num_classes=num_class, imagenet=imagenet).cuda()
  else:
    assert False

  process_group = torch.distributed.new_group(list(range(world_size)))
  model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

  model = model.cuda()
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
  decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

  start_epoch = args.start_epoch

  if args.checkpoint != '':
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
    elif 'P_state' in checkpoint:
      state_dict = checkpoint['P_state']
    else:
      state_dict = checkpoint

    if args.cvt_state_dict:
      state_dict = cvt_state_dict(state_dict, args, model.module.fc.in_features, num_class)

    model.load_state_dict(state_dict)
    fix_model(model, log)
    log.info('read checkpoint {}'.format(args.checkpoint))

  elif args.resume:
    checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
    if 'state_dict' in checkpoint:
      model.load_state_dict(checkpoint['state_dict'])
    else:
      model.load_state_dict(checkpoint)

  if args.resume:
    if 'epoch' in checkpoint and 'optim' in checkpoint:
      start_epoch = checkpoint['epoch']
      optimizer.load_state_dict(checkpoint['optim'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
    else:
      log.info("cannot resume since lack of files")
      assert False

  ta = []
  best_prec1 = 0

  if not args.test_only:
    for epoch in range(start_epoch + 1, args.epochs + 1):
      log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
      train_sampler.set_epoch(epoch)

      # adversarial training
      train(args, model, device, train_loader, optimizer, epoch, log, world_size=world_size, scheduler=scheduler)

      # adjust learning rate for SGD
      scheduler.step()

      if epoch % args.test_freq == 0:
        # evaluation on natural examples
        log.info('================================================================')
        _, _, top5_vali_tacc = eval_test(model, device, val_loader, log, world_size, prefix='vali', num_class=num_class)
        ta.append(top5_vali_tacc)
        log.info('================================================================')

        if args.local_rank == 0:
          # save checkpoint
          torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim': optimizer.state_dict(),
            'best_prec1': best_prec1,
            'scheduler': scheduler.state_dict(),
          }, os.path.join(model_dir, 'model.pt'))

          if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_prec1': best_prec1,
            }, os.path.join(model_dir, 'model_{}.pt'.format(epoch)))

          is_best = top5_vali_tacc > best_prec1
          best_prec1 = max(top5_vali_tacc, best_prec1)

          if is_best:
            torch.save({
              'epoch': epoch,
              'state_dict': model.state_dict(),
              'optim': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'best_prec1': best_prec1,
            }, os.path.join(model_dir, 'best_model.pt'))
    torch.distributed.barrier()

  checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
  model.load_state_dict(checkpoint['state_dict'])
  _, _, test_top5_tacc = eval_test(model, device, test_loader, log, world_size, num_class=num_class)
  log.info("On the best_model, test top5 tacc is {}".format(test_top5_tacc))


def cvt_state_dict(state_dict, args, in_features, num_class):
  # deal with adv bn
  state_dict_new = copy.deepcopy(state_dict)

  if args.bnNameCnt >= 0:
    for name, item in state_dict.items():
      if 'bn' in name:
        assert 'bn_list' in name
        state_dict_new[name.replace('.bn_list.{}'.format(args.bnNameCnt), '')] = item

  name_to_del = []
  for name, item in state_dict_new.items():
    if 'bn_list' in name:
      name_to_del.append(name)
    if 'fc' in name:
      name_to_del.append(name)
  for name in np.unique(name_to_del):
    del state_dict_new[name]

  keys = list(state_dict_new.keys())[:]
  name_to_del = []
  for name in keys:
    if 'downsample.conv' in name:
      state_dict_new[name.replace('downsample.conv', 'downsample.0')] = state_dict_new[name]
      name_to_del.append(name)
    if 'downsample.bn' in name:
      state_dict_new[name.replace('downsample.bn', 'downsample.1')] = state_dict_new[name]
      name_to_del.append(name)
  for name in np.unique(name_to_del):
    del state_dict_new[name]

  # zero init fc
  # set_trace()

  if in_features == 0:
    pass
  else:
    state_dict_new['module.fc.weight'] = torch.zeros(num_class, in_features).normal_(mean=0.0, std=0.01).to(state_dict_new['module.conv1.weight'].device)
    state_dict_new['module.fc.bias'] = torch.zeros(num_class).to(state_dict_new['module.conv1.weight'].device)

  return state_dict_new


if __name__ == '__main__':
  main()
