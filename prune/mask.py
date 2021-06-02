# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import numpy as np
import torch
from collections import OrderedDict

from pdb import set_trace


class Mask(object):
    def __init__(self, model, no_reset=False):
        super(Mask, self).__init__()
        self.model = model
        if not no_reset:
            self.reset()

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in prunableTensors]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

    def magnitudePruning(self, magnitudePruneFraction, randomPruneFraction=0):
        weights = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights.append(module.weight.clone().cpu().detach().numpy())

        # only support one time pruning
        self.reset()
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        number_of_remaining_weights = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors])).cpu().numpy()
        number_of_weights_to_prune_magnitude = np.ceil(magnitudePruneFraction * number_of_remaining_weights).astype(int)
        number_of_weights_to_prune_random = np.ceil(randomPruneFraction * number_of_remaining_weights).astype(int)
        random_prune_prob = number_of_weights_to_prune_random / (number_of_remaining_weights - number_of_weights_to_prune_magnitude)

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v.flatten() for v in weights])
        threshold = np.sort(np.abs(weight_vector))[min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

        # apply the mask
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = (torch.abs(module.weight) >= threshold).float()
                # random weights been pruned
                module.prune_mask[torch.rand_like(module.prune_mask) < random_prune_prob] = 0

    def reset(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = torch.ones_like(module.weight)


def save_mask(epoch, model, filename):
    pruneMask = OrderedDict()

    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            pruneMask[name] = module.prune_mask.cpu().type(torch.bool)

    torch.save({"epoch": epoch, "pruneMask": pruneMask}, filename)


def load_mask(model, state_dict, device):
    # set_trace()
    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            module.prune_mask.data = state_dict[name].to(device).float()

    return model


if __name__ == "__main__":
    from models.resnet_prune import prune_resnet18
    net = prune_resnet18().cuda()
    image = torch.rand(3, 224, 224).cuda()
    mask = Mask(net)

    for rate in (0, 0.5):
        # prune 0%
        # mask.magnitudePruning(0)
        mask.magnitudePruning(rate)
        net.set_prune_flag(True)
        a = net(image)
        print("prune, density is {}, avg is {}".format(mask.density, a.mean()))
        # net.set_prune_flag(False)
        mask.magnitudePruning(rate+0.1)
        b = net(image)
        print("no prune, density is {}, avg is {}".format(mask.density, b.mean()))
        (a + b).mean().backward()
