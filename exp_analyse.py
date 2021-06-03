import re
import os
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
from data.cifar10 import CustomCIFAR10
from data.cifar100 import CustomCIFAR100
import argparse

# import torch
from pdb import set_trace


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment summary parser')
    parser.add_argument('--dataset', default='cifar100', type=str)
    parser.add_argument('--LT', action='store_true', help='if use long tail distribution')
    parser.add_argument('--fewShot', action='store_true', help='if use fewShot')
    parser.add_argument('--prune', action='store_true', help='if use pruning')
    return parser.parse_args()


def getStatisticsFromTxt(txtName, num_class=1000):
    statistics = [0 for _ in range(num_class)]
    with open(txtName, 'r') as f:
        lines = f.readlines()
    for line in lines:
        s = re.search(r" ([0-9]+)$", line)
        if s is not None:
            statistics[int(s[1])] += 1
    return statistics


def getStatistics_cifar10(trainSplit, root="../../data"):
    train_idx = list(np.load('split/{}'.format(trainSplit)))
    train_datasets = CustomCIFAR10(train_idx, root=root, train=True, transform=None, download=True)
    statistics = train_datasets.idxsNumPerClass
    return statistics


def getStatistics_cifar100(trainSplit, root="../../data"):
    train_idx = list(np.load('split/{}'.format(trainSplit)))
    train_datasets = CustomCIFAR100(train_idx, root=root, train=True, transform=None, download=True)
    statistics = train_datasets.idxsNumPerClass
    return statistics


def getAccAsimclr(saveDir, exp):
    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return -1
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    bestAcc = -1
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^On the best_model, test tacc is ([0-9]+\.[0-9]+)$", line)
        if groups:
            bestAcc = float(groups[1])

    return bestAcc


def getClassWiseAccAsimclr(saveDir, exp, classnum=10):
    """
    :param line:
    :param save_list:
    :return:
    """
    strList = ""
    for i in range(classnum):
        strList += " ([0-9]+\.[0-9]+)"

    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return []
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    save_list = []
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^Each class acc is{}".format(strList), line)
        if groups:
            for i in range(classnum):
                save_list.append(float(groups[i+1]))

    return save_list


def getClassWiseAccImagenet(saveDir, exp, classnum=1000):
    """
    :param line:
    :param save_list:
    :return:
    """
    strList = ""
    for i in range(classnum):
        strList += " ([0-9]+\.[0-9]+)"

    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return []
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    save_list = []
    for line in lines[-5:]:
        # set_trace()
        groups = re.match("^acc per class is{}".format(strList), line)
        if groups:
            for i in range(classnum):
                save_list.append(float(groups[i+1]))

    return save_list


def getAccImagenet(saveDir, exp):
    path = osp.join(saveDir, exp, 'log.txt')
    if not osp.isfile(path):
        return -1
    with open(path, 'r') as file:
        lines = file.read().splitlines()

    bestAcc = -1
    for line in lines[-20:]:
        # set_trace()
        groups = re.match("^On the best_model, test top5 tacc is ([0-9]+\.[0-9]+)", line)
        if groups:
            bestAcc = float(groups[1])

    return bestAcc


def autoSummaryExpRes(saveDir, exps, prefix, dataset='cifar10',
                      noReturnAvg=False, returnValue=False, getInfo="Asimclr", group=3, noGroup=False):
    '''
    Args:
        saveDir: str, path to save
        exps: list of tuple: (exp, index_split)
        prefix: display prefix
        dataset: which dataset

    Returns:
    '''
    accList = []
    fullVarianceList = []
    GroupVarienceList = []
    majorAccList = []
    moderateAccList = []
    minorAccList = []
    top5AccList = []
    low5AccList = []

    for exp, index_split in exps:
        if getInfo == "Asimclr":
            bestAcc = getAccAsimclr(saveDir, exp)
        elif getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            bestAcc = getAccImagenet(saveDir, exp)
        if bestAcc < 0:
            print("miss exp {}".format(exp))
            continue

        # get major moderate minor class accuracy
        if dataset == 'cifar10':
            currentStatistics = getStatistics_cifar10('cifar10_imbSub_with_subsets/split{}_D_i.npy'.format(index_split))
        elif dataset == 'cifar100':
            currentStatistics = getStatistics_cifar100('cifar100_imbSub_with_subsets/cifar100_split{}_D_i.npy'.format(index_split))
        elif dataset == 'Imagenet':
            currentStatistics = np.array(getStatisticsFromTxt('split/ImageNet_LT/imageNet_LT_exp_train.txt'))
        elif dataset == 'Imagenet-100':
            currentStatistics = np.array(getStatisticsFromTxt('split/imagenet-100/imageNet_100_LT_train.txt', num_class=100))
        else:
            assert False

        if getInfo == "Asimclr":
            classWiseAcc = getClassWiseAccAsimclr(saveDir, exp, classnum=len(currentStatistics))
        elif getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            # set_trace()
            classWiseAcc = getClassWiseAccImagenet(saveDir, exp, classnum=len(currentStatistics))
        else:
            assert False

        # set_trace()
        if not classWiseAcc:
            print("miss classwise acc for {}".format(exp))
            assert False

        sortIdx = np.argsort(currentStatistics)
        idxsMajor = sortIdx[len(currentStatistics) // 3 * 2:]
        idxsModerate = sortIdx[len(currentStatistics) // 3 * 1: len(currentStatistics) // 3 * 2]
        idxsMinor = sortIdx[: len(currentStatistics) // 3 * 1]

        # set_trace()

        classWiseAcc = np.array(classWiseAcc)
        if getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            classWiseAcc = classWiseAcc * 100
            print("classWiseAcc is {}".format(classWiseAcc))

        bestAcc = np.mean(classWiseAcc)
        majorAcc = np.mean(classWiseAcc[idxsMajor])
        moderateAcc = np.mean(classWiseAcc[idxsModerate])
        minorAcc = np.mean(classWiseAcc[idxsMinor])

        if getInfo == "Imagenet" or getInfo == "Imagenet-100" or getInfo == "Places":
            idxsMany = np.nonzero(currentStatistics > 100)[0]
            idxsMedium = np.nonzero((100 >= currentStatistics) & (currentStatistics >= 20))[0]
            idxsFew = np.nonzero(currentStatistics < 20)[0]
            majorAcc = np.mean(classWiseAcc[idxsMany])
            moderateAcc = np.mean(classWiseAcc[idxsMedium])
            minorAcc = np.mean(classWiseAcc[idxsFew])

        accList.append(bestAcc)
        majorAccList.append(majorAcc)
        moderateAccList.append(moderateAcc)
        minorAccList.append(minorAcc)
        # balancenessList.append(imbalance_metric(classWiseAcc / 100, sigma=1))
        # print("classWiseAcc is {}".format(classWiseAcc))
        fullVarianceList.append(np.std(classWiseAcc / 100))
        GroupVarienceList.append(np.std(np.array([majorAcc, moderateAcc, minorAcc]) / 100))

        if group > 3:
            assert len(classWiseAcc) % group == 0
            group_idx_list = [sortIdx[len(currentStatistics) // group * cnt: len(currentStatistics) // group * (cnt + 1)] \
                              for cnt in range(0, group)]
            group_accs = [np.mean(classWiseAcc[group_idx_list[cnt]]) for cnt in range(0, group)]
            outputStr = "{}: group accs are".format(prefix)
            for acc in group_accs:
                outputStr += " {:.02f}".format(acc)
            print(outputStr)

    if returnValue:
        return accList, majorAccList, moderateAccList, minorAccList
    else:
        if noReturnAvg:
            outputStr = "{}: accs are".format(prefix)
            for acc in accList:
                outputStr += " {:.02f}".format(acc)
            print(outputStr)
            if not noGroup:
                outputStr = "{}: majorAccs are".format(prefix)
                for acc in majorAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: moderateAccs are".format(prefix)
                for acc in moderateAccList:
                    outputStr += " {:.02f}".format(acc)
                print(outputStr)
                outputStr = "{}: minorAccs are".format(prefix)
                for acc in minorAccList:
                    outputStr += " {:.02f}".format(acc)
            print(outputStr)
        else:
            print("{}: acc is {:.02f}+-{:.02f}".format(prefix, np.mean(accList), np.std(accList)))
            if not noGroup:
                print("{}: vaiance is {:.04f}+-{:.04f}".format(prefix, np.mean(fullVarianceList), np.std(fullVarianceList)))
                print("{}: GroupBalancenessList is {:.04f}+-{:.04f}".format(prefix, np.mean(GroupVarienceList), np.std(GroupVarienceList)))
                print("{}: major acc is {:.02f}+-{:.02f}".format(prefix, np.mean(majorAccList), np.std(majorAccList)))
                print("{}: moderate acc is {:.02f}+-{:.02f}".format(prefix, np.mean(moderateAccList), np.std(moderateAccList)))
                print("{}: minor acc is {:.02f}+-{:.02f}".format(prefix, np.mean(minorAccList), np.std(minorAccList)))


def summaryCifar10(longTailDataset=True, prune=False, fewShot=False):
    epoch = 2000
    seed = 10
    pruningPercent = 0.9
    saveDir = "checkpoints_tune"

    if not longTailDataset:
        subset = "split{}_D_b"
        dataset = "D_b"
    elif longTailDataset:
        subset = "split{}_D_i"
        dataset = "D_i"

    if prune:
        pretrain_name = "res18_scheduling_sgd_temp0.2_wd1e-4_lr0.5_b512_twolayerProj_epoch{}_{}_newNT_s{}_pruneP{}DualBN".format(epoch, subset, seed, pruningPercent)
    else:
        pretrain_name = "res18_scheduling_sgd_temp0.2_wd1e-4_lr0.5_b512_twolayerProj_epoch{}_{}_newNT_s{}".format(epoch, subset, seed)

    if fewShot:
        tuneSet = "split{}_S_b"
        exps = [["{}__{}_f2layer4_d40d60_wd0_lr30_freezeBN".format(
            pretrain_name.format(index_split), tuneSet.format(index_split)), index_split] for index_split in range(1, 6)]
    else:
        exps = [["{}__f2layer4_d10d20_wd0_lr30_freezeBN".format(pretrain_name.format(index_split)), index_split] for index_split in range(1, 6)]

    print("exp[0] is {}".format(exps[0]))
    autoSummaryExpRes(saveDir, exps, "cifar10 Dataset:{} prune:{} {}".format(dataset, prune, "fewShot" if fewShot else "fullShot"), dataset='cifar10')


def summaryCifar100(longTailDataset=True, prune=False, fewShot=False):
    epoch = 2000
    seed = 10
    pruningPercent = 0.9
    saveDir = "checkpoints_cifar100_tune"

    if not longTailDataset:
        subset = "cifar100_split{}_D_b"
        dataset="D_b"
    elif longTailDataset:
        subset = "cifar100_split{}_D_i"
        dataset = "D_i"

    if prune:
        pretrain_name = "res18_cifar100_scheduling_sgd_temp0.2_wd1e-4_lr0.5_b512_o128_twolayerProj_epoch{}_{}_newNT_s{}_pruneP{}DualBN".format(epoch, subset, seed, pruningPercent)
    else:
        pretrain_name = "res18_cifar100_scheduling_sgd_temp0.2_wd1e-4_lr0.5_b512_o128_twolayerProj_epoch{}_{}_newNT_s{}".format(epoch, subset, seed)

    if fewShot:
        tuneSet = "cifar100_split{}_S_b"
        exps = [["{}__{}_f2layer4_d40d60_wd0_lr30_freezeBN".format(
            pretrain_name.format(index_split), tuneSet.format(index_split)), index_split] for index_split in range(1, 6)]
    else:
        exps = [["{}__f2layer4_d10d20_wd0_lr30_freezeBN".format(pretrain_name.format(index_split)), index_split] for index_split in range(1, 6)]

    print("exp[0] is {}".format(exps[0]))
    autoSummaryExpRes(saveDir, exps, "cifar100 Dataset:{} prune:{} {}".format(dataset, prune, "fewShot" if fewShot else "fullShot"), dataset='cifar100')


def summaryIN100(longTailDataset=True, prune=False, fewShot=False):
    saveDir = "checkpoints_imagenet_tune"

    if longTailDataset:
        exp_name = "imageNet_100_LT_train_res50_scheduling_sgd_lr0.5_temp0.2_epoch500_batch256"
    else:
        assert not prune
        exp_name = "imageNet_100_BL_train_res50_scheduling_sgd_lr0.5_temp0.2_epoch500_batch256"

    if prune:
        exp_name = "imageNet_100_LT_train_res50_scheduling_sgd_lr0.5_temp0.3_epoch500_batch256_pruneP0.3DualBN"

    if fewShot:
        exp_name = "{}__{}".format(exp_name, "imageNet_100_sub_balance_train_0.01_lr30_fix2layer4_wd0_epoch30_b512_d10d20")
    else:
        exp_name = "{}__{}".format(exp_name, "lr30_wd0_epoch30_b512_d10d20_s1")

    exps = [[exp_name, 0]]
    # splitSystem = "imbSub" if "SC" not in exp_name else "imbSub_SC"
    splitSystem = "imbSub"
    autoSummaryExpRes(saveDir, exps, "{} ".format(splitSystem), getInfo='Imagenet-100', dataset='Imagenet-100')


def summaryIN(longTailDataset=True, fewShot=False):
    saveDir = "checkpoints_imagenet_tune"

    if longTailDataset:
        exp_name = "imageNet_LT_exp_train_res50_scheduling_sgd_lr0.5_temp0.2_epoch500_batch256"
    else:
        exp_name = "imageNet_BL_exp_train_res50_scheduling_sgd_lr0.5_temp0.2_epoch500_batch256"

    if fewShot:
        exp_name = "{}__{}".format(exp_name, "imageNet_sub_balance_train_0.01_lr30_fix2layer4_wd0_epoch30_b512_d10d20")
    else:
        exp_name = "{}__{}".format(exp_name, "lr30_wd0_epoch30_b512_d10d20")

    exps = [[exp_name, 0]]
    # splitSystem = "imbSub" if "SC" not in exp_name else "imbSub_SC"
    splitSystem = "imbSub"
    autoSummaryExpRes(saveDir, exps, "{} ".format(splitSystem), getInfo='Imagenet', dataset='Imagenet')


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "cifar10":
        summaryCifar10(args.LT, args.prune, args.fewShot)
    elif args.dataset == "cifar100":
        summaryCifar100(args.LT, args.prune, args.fewShot)
    elif args.dataset == "imagenet100":
        summaryIN100(args.LT, args.prune, args.fewShot)
    elif args.dataset == "imagenet":
        summaryIN(args.LT, args.fewShot)
    else:
        raise ValueError("dataset of {} is not supported, supported datasets includes [cifar10, cifar100, imagenet100, imagenet]".format(args.dataset))