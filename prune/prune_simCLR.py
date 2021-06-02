import torch.optim
import torch.utils.data
from utils import *

import torch.distributed as dist
from .mask import Mask


def gatherFeatures(features, local_rank, world_size):
    features_list = [torch.zeros_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(features_list, features)
    features_list[local_rank] = features
    features = torch.cat(features_list)
    return features


def train_prune(train_loader, model, optimizer, scheduler, epoch, log, local_rank, rank, world_size, args=None):
    pruneMask = Mask(model)
    prunePercent = args.prune_percent
    randomPrunePercent = args.random_prune_percent
    magnitudePrunePercent = prunePercent - randomPrunePercent

    log.info("current prune percent is {}".format(prunePercent))
    if randomPrunePercent > 0:
        log.info("random prune percent is {}".format(randomPrunePercent))
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()

    end = time.time()

    # prune every epoch
    pruneMask.magnitudePruning(magnitudePrunePercent, randomPrunePercent)

    for i, (inputs) in enumerate(train_loader):

        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()

        inputs = inputs.cuda(non_blocking=True)

        inputs_1 = inputs[:, 0, ...]
        inputs_2 = inputs[:, 1, ...]

        # As pytorch does not support run the same module twice without backward under distribution training,
        # we utilize the chain rule to make contrasting different models happen
        model.train()
        optimizer.zero_grad()

        # calculate the grad for non-pruned network
        with torch.no_grad():
            # branch with pruned network
            model.module.set_prune_flag(True)
            features_2 = model(inputs_2)
            features_2_noGrad = gatherFeatures(features_2, rank, world_size).detach()
        model.module.set_prune_flag(False)
        features_1 = model(inputs_1)
        features_1 = gatherFeatures(features_1, rank, world_size)

        loss = nt_xent(features_1, features2=features_2_noGrad, t=args.temperature)

        # normalize the loss
        loss = loss * world_size
        loss.backward()

        loss_val = float(loss.detach().cpu() / world_size)

        # calculate the grad for pruned network
        features_1_no_grad = features_1.detach()
        model.module.set_prune_flag(True)
        features_2 = model(inputs_2)
        features_2 = gatherFeatures(features_2, rank, world_size)

        loss = nt_xent(features_1_no_grad, features2=features_2, t=args.temperature)

        loss = loss * world_size
        loss.backward()

        optimizer.step()

        losses.update(loss_val, inputs.shape[0])

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

