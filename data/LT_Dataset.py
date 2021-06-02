import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import random
import numpy as np
import pickle
from matplotlib import pyplot as plt
from pdb import set_trace
# from data.utils import _gaussian_blur
import torch.distributed as dist
import torchvision.transforms as transforms


class LT_Dataset(Dataset):

  def __init__(self, root, txt, transform=None, returnPath=False):
    self.img_path = []
    self.labels = []
    self.root = root
    self.transform = transform
    self.returnPath = returnPath
    self.txt = txt

    with open(txt) as f:
      for line in f:
        self.img_path.append(os.path.join(root, line.split()[0]))
        self.labels.append(int(line.split()[1]))

    self.targets = self.labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):

    path = self.img_path[index]
    label = self.labels[index]

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    if self.transform is not None:
      sample = self.transform(sample)

    if not self.returnPath:
      return sample, label, index
    else:
      return sample, label, index, path.replace(self.root, '')


def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo)
  return dict


class Unsupervised_LT_Dataset(LT_Dataset):
  def __init__(self, returnIdx=False, returnLabel=False, **kwds):
    super().__init__(**kwds)
    self.returnIdx = returnIdx
    self.returnLabel = returnLabel

  def __getitem__(self, index):
    path = self.img_path[index]
    label = self.labels[index]

    if not os.path.isfile(path):
      path = path + ".gz"

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    samples = [self.transform(sample), self.transform(sample)]
    if self.returnIdx and (not self.returnLabel):
      return torch.stack(samples), index
    elif (not self.returnIdx) and self.returnLabel:
      return torch.stack(samples), label
    elif self.returnIdx and self.returnLabel:
      return torch.stack(samples), label, index
    else:
      return torch.stack(samples)


class LT_Dataset_Meta(Dataset):

  def __init__(self, root, txt, classRange, transform=None, returnPath=False):
    self.img_path = []
    self.labels = []
    self.root = root
    self.transform = transform
    self.returnPath = returnPath

    self.txt = txt
    with open(txt) as f:
      for line in f:
        self.img_path.append(os.path.join(root, line.split()[0]))
        self.labels.append(int(line.split()[1]))

    self.setClassRange(classRange)

  def setClassRange(self, classRange):
    self.epochList = []
    self.classRange = classRange
    for idx, label in enumerate(self.labels):
      if label in self.classRange:
        self.epochList.append(idx)

  def __len__(self):
    return len(self.epochList)

  def __getitem__(self, index):
    sampleIdx = self.epochList[index]

    path = self.img_path[sampleIdx]
    label = self.labels[sampleIdx]
    label = self.classRange.index(label)

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')

    if self.transform is not None:
      sample = self.transform(sample)

    if not self.returnPath:
      return sample, label, index
    else:
      return sample, label, index, path.replace(self.root, '')


