import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from models.resnet import resnet18, resnet10, resnet50, resnet101, resnet152, wide_resnet50_2
from models.resnet_prune import prune_resnet10, prune_resnet18, prune_resnet34, prune_resnet50, prune_resnet101, prune_resnet152
from models.resnet_prune_multibn import prune_resnet10_dual, prune_resnet18_dual, prune_resnet34_dual, prune_resnet50_dual, prune_resnet101_dual, prune_resnet152_dual
from models.resnet_s_cifar import resnet32_s, resnet32_BBN_s
from models.resnet_s_cifar_prune_multibn import resnet32_prune_dual_s, resnet32_BBN_prune_dual_s
from models.ride_resnet_s_cifar_prune_multibn import resnet32 as resnet32_s_ride_prune_dual
from models.ride_resnet_s_cifar import resnet32 as resnet32_s_ride
from utils import *
import torchvision.transforms as transforms
import torch.distributed as dist

import numpy as np
import copy

from data.cifar10 import CustomCIFAR10
from data.cifar100 import CustomCIFAR100
from data.LT_Dataset import Unsupervised_LT_Dataset
from optimizer.lars import LARS
from data.augmentation import GaussianBlur

from prune.prune_simCLR import train_prune
from prune.mask import Mask, save_mask

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='/hdd3/ziyu/SS_imbalance/checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('--data', type=str, default='', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset, [imagenet-LT, imagenet-100, places, cifar, cifar100]')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_freq', default=100, type=int, help='save frequency /epoch')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', default=False, type=bool, help='if resume training')
parser.add_argument('--optimizer', default='lars', type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--model', default='res18', type=str, help='model type')
parser.add_argument('--temperature', default=0.5, type=float, help='nt_xent temperature')
parser.add_argument('--output_ch', default=512, type=int, help='proj head output feature number')

parser.add_argument('--trainSplit', type=str, default='trainIdxList.npy', help="train split")
parser.add_argument('--imagenetCustomSplit', type=str, default='', help="imagenet custom split")

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')

parser.add_argument('--strength', default=1.0, type=float, help='cifar augmentation, color jitter strength')
parser.add_argument('--resizeLower', default=0.1, type=float, help='resize smallest size')

parser.add_argument('--testContrastiveAcc', action='store_true', help="test contrastive acc")
parser.add_argument('--testContrastiveAccTest', action='store_true', help="test contrastive acc in test set")

# contrast with pruned model
parser.add_argument('--prune', action='store_true', help="if contrasting with pruned model")
parser.add_argument('--prune_percent', type=float, default=0.3, help="whole prune percentage")
parser.add_argument('--random_prune_percent', type=float, default=0, help="random prune percentage")
parser.add_argument('--prune_dual_bn', action='store_true', help="if employing dual bn during pruning")

# save mask
parser.add_argument('--mask_save_freq', type=int, default=100, help="freq for saving mask")


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0

    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))

    return lr


def main():
    global args
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")
    torch.cuda.set_device(args.local_rank)
    
    rank = torch.distributed.get_rank()
    logName = "log.txt"

    log = logger(path=save_dir, local_rank=rank, log_name=logName)
    log.info(str(args))

    setup_seed(args.seed + rank)
    
    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))
    print("rank is {}, world size is {}".format(rank, world_size))

    assert args.batch_size % world_size == 0
    batch_size = args.batch_size // world_size

    # define model
    if args.dataset == 'imagenet-LT' or args.dataset == 'imagenet-100' or args.dataset == 'places':
        imagenet = True
    elif args.dataset == 'cifar' or args.dataset == 'cifar100':
        imagenet = False
    else:
        assert False

    if 'imagenet' in args.dataset:
        num_class = 1000
        if 'imagenet-100' in args.dataset:
            num_class = 100
    elif args.dataset == 'cifar':
        num_class = 10
    elif args.dataset == 'cifar100':
        num_class = 100
    else:
        assert False

    if args.model == 'res18':
        model = resnet18(pretrained=False, imagenet=imagenet, num_classes=num_class)
        if args.prune:
            model = prune_resnet18(pretrained=False, imagenet=imagenet, num_classes=num_class)
            if args.prune_dual_bn:
                model = prune_resnet18_dual(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res32_s':
        assert not imagenet
        model = resnet32_s(num_classes=num_class)
        if args.prune:
            assert args.prune_dual_bn
            model = resnet32_prune_dual_s(num_classes=num_class)
    elif args.model == 'res32_s_ride':
        assert not imagenet
        model = resnet32_s_ride(num_classes=num_class)
        if args.prune:
            assert args.prune_dual_bn, "no implementation"
            model = resnet32_s_ride_prune_dual(num_classes=num_class)
    elif args.model == 'res32_bbn_s':
        assert not imagenet
        model = resnet32_BBN_s(num_classes=num_class)
        if args.prune:
            assert args.prune_dual_bn, "no implementation"
            model = resnet32_BBN_prune_dual_s(num_classes=num_class)
    elif args.model == 'res10':
        model = resnet10(pretrained=False, imagenet=imagenet, num_classes=num_class)
        if args.prune:
            model = prune_resnet10(pretrained=False, imagenet=imagenet, num_classes=num_class)
            if args.prune_dual_bn:
                model = prune_resnet10_dual(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res50':
        model = resnet50(pretrained=False, imagenet=imagenet, num_classes=num_class)
        if args.prune:
            model = prune_resnet50(pretrained=False, imagenet=imagenet, num_classes=num_class)
            if args.prune_dual_bn:
                model = prune_resnet50_dual(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res101':
        model = resnet101(pretrained=False, imagenet=imagenet, num_classes=num_class)
        if args.prune:
            model = prune_resnet101(pretrained=False, imagenet=imagenet, num_classes=num_class)
            if args.prune_dual_bn:
                model = prune_resnet101_dual(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res152':
        model = resnet152(pretrained=False, imagenet=imagenet, num_classes=num_class)
        if args.prune:
            model = prune_resnet152(pretrained=False, imagenet=imagenet, num_classes=num_class)
            if args.prune_dual_bn:
                model = prune_resnet152_dual(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res50w2':
        model = wide_resnet50_2(pretrained=False, imagenet=imagenet, num_classes=num_class)
        if args.prune:
            assert False, "no implementation"
    else:
        assert False, "no such model"

    if not args.prune:
        assert not args.prune_dual_bn

    if model.fc is None:
        # hard coding here, for ride resent
        ch = 192
    else:
        ch = model.fc.in_features

    if args.prune_dual_bn:
        from models.resnet_prune_multibn import proj_head
        model.fc = proj_head(ch, args.output_ch)
    else:
        from models.utils import proj_head
        model.fc = proj_head(ch, args.output_ch)

    model.cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    cudnn.benchmark = True

    if args.dataset == "cifar100" or args.dataset == "cifar":
        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * args.strength, 0.4 * args.strength,
                                                                          0.4 * args.strength, 0.1 * args.strength)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(args.resizeLower, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.ToTensor(),
        ])

        tfs_test = transforms.Compose([
              transforms.ToTensor(),
          ])

    elif args.dataset == "imagenet-LT" or args.dataset == 'imagenet-100':

        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)

        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
        ])

        tfs_test = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          ])
    else:
        assert False

    # dataset process
    if args.dataset == "cifar":
        # the data distribution
        if args.data == '':
            root = '../../data'
        else:
            root = args.data

        train_idx = list(np.load('split/{}'.format(args.trainSplit)))
        train_datasets = CustomCIFAR10(train_idx, root=root, train=True, transform=tfs_train, download=True)

    elif args.dataset == "cifar100":
        assert not args.testContrastiveAccTest
        # the data distribution
        if args.data == '':
            root = '../../data'
        else:
            root = args.data

        assert 'cifar100' in args.trainSplit
        train_idx = list(np.load('split/{}'.format(args.trainSplit)))


        train_datasets = CustomCIFAR100(train_idx, root=root, train=True, transform=tfs_train, download=True)

    elif args.dataset == "imagenet-LT" or args.dataset == 'imagenet-FULL' or args.dataset == 'imagenet-100':
        if args.dataset == 'imagenet-100':
            txt = "split/imagenet-100/ImageNet_100_train.txt"
            if args.imagenetCustomSplit != '':
                txt = "split/imagenet-100/{}.txt".format(args.imagenetCustomSplit)
            print("use imagenet-100 {}".format(args.imagenetCustomSplit))
        else:
            if args.imagenetCustomSplit != '':
                txt = "split/ImageNet_LT/{}.txt".format(args.imagenetCustomSplit)
                print("use {}".format(txt))
            elif args.dataset == "imagenet-LT":
                print("use imagenet long tail")
                txt = "split/ImageNet_LT/ImageNet_LT_train.txt"
            else:
                print("use imagenet full set")
                txt = "split/ImageNet_LT/ImageNet_train.txt"

        root = getImagenetRoot(args.data)

        train_datasets = Unsupervised_LT_Dataset(root=root, txt=txt, transform=tfs_train)

        class_stat = [0 for _ in range(num_class)]
        for lbl in train_datasets.labels:
            class_stat[lbl] += 1
        log.info("class distribution in training set is {}".format(class_stat))
    else:
        assert False

    shuffle = True
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=shuffle)
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=args.num_workers,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=False)

    if args.dataset == "cifar" or args.dataset == "cifar100":
        root = args.data
        if os.path.isdir(root):
            pass
        elif os.path.isdir('../../data'):
            root = '../../data'

        if args.dataset == "cifar":
            val_train_datasets = datasets.CIFAR10(root=root, train=True, transform=tfs_test, download=True)
        else:
            val_train_datasets = datasets.CIFAR100(root=root, train=True, transform=tfs_test, download=True)
        val_train_sampler = SubsetRandomSampler(train_idx)
        val_train_loader = torch.utils.data.DataLoader(val_train_datasets, batch_size=batch_size, sampler=val_train_sampler)

        class_stat = [0 for _ in range(num_class)]
        for imgs, targets in val_train_loader:
            for target in targets:
                class_stat[target] += 1
        log.info("class distribution in training set is {}".format(class_stat))


    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs * len(train_loader) * 10, ], gamma=1)
    elif args.scheduler == 'cosine':
        training_iters = args.epochs * len(train_loader)
        warm_up_iters = 10 * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    training_iters,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=warm_up_iters)
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'), map_location="cuda")
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'epoch' in checkpoint and 'optim' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optim'])

                for i in range((start_epoch - 1) * len(train_loader)):
                    scheduler.step()
                log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            else:
                log.info("cannot resume since lack of files")
                assert False

    for epoch in range(start_epoch, args.epochs + 1):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_sampler.set_epoch(epoch)

        if args.prune:
            train_prune(train_loader, model, optimizer, scheduler, epoch, log, args.local_rank, rank, world_size, args=args)
        else:
            train(train_loader, model, optimizer, scheduler, epoch, log, args.local_rank, rank, world_size, args=args)

        if rank == 0:
            if imagenet:
                save_model_freq = 1
            else:
                save_model_freq = 2

            if epoch % save_model_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model.pt'))

            if epoch % args.save_freq == 0 and epoch > 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))
                
            if epoch % args.mask_save_freq == 0 and args.prune:
                save_mask(epoch=epoch, model=model, filename=os.path.join(save_dir, 'mask_{}.pt'.format(epoch)))


def train(train_loader, model, optimizer, scheduler, epoch, log, local_rank, rank, world_size, args=None):
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()

    for i, (inputs) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        d = inputs.size()
        # print("inputs origin shape is {}".format(d))
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda(non_blocking=True)

        model.train()
        features = model(inputs)

        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(features_list, features)
        features_list[rank] = features
        features = torch.cat(features_list)

        loss = nt_xent(features, t=args.temperature)
        # normalize the loss
        loss = loss * world_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(float(loss.detach().cpu() / world_size), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        # torch.cuda.empty_cache()
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                     'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))

    return losses.avg


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()


