import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools


def to_var(x, requires_grad=True):
  if torch.cuda.is_available():
    x = x.cuda()
  return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
  # adopted from: Adrien Ecoffet https://github.com/AdrienLE
  def params(self):
    for name, param in self.named_params(self):
      yield param

  def named_leaves(self):
    return []

  def named_submodules(self):
    return []

  def named_params(self, curr_module=None, memo=None, prefix=''):
    if memo is None:
      memo = set()

    if hasattr(curr_module, 'named_leaves'):
      for name, p in curr_module.named_leaves():
        if p is not None and p not in memo:
          memo.add(p)
          yield prefix + ('.' if prefix else '') + name, p
    else:
      for name, p in curr_module._parameters.items():
        if p is not None and p not in memo:
          memo.add(p)
          yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():
      submodule_prefix = prefix + ('.' if prefix else '') + mname
      for name, p in self.named_params(module, memo, submodule_prefix):
        yield name, p

  def update_params(self, lr_inner, first_order=False, source_params=None, detach=False, lars_update=False):
    if source_params is not None:
      for tgt, src in zip(self.named_params(self), source_params):
        name_t, param_t = tgt
        # name_s, param_s = src
        # grad = param_s.grad
        # name_s, param_s = src
        grad = src
        if first_order:
          grad = to_var(grad.detach().data)

        if lars_update:
          weight_norm = torch.norm(param_t.data)
          grad_norm = torch.norm(grad)

          # Global LR computed on polynomial decay schedule
          # decay = (1 - float(epoch) / max_epoch) ** 2
          global_lr = lr_inner

          eta = 0.001
          # Compute local learning rate for this layer
          local_lr = eta * weight_norm / \
                     (grad_norm)

          # Update the momentum term
          actual_lr = local_lr * global_lr
          lr_inner = actual_lr

        tmp = param_t - lr_inner * grad
        self.set_param(self, name_t, tmp)
    else:

      for name, param in self.named_params(self):
        if not detach:
          grad = param.grad
          if first_order:
            grad = to_var(grad.detach().data)
          tmp = param - lr_inner * grad
          self.set_param(self, name, tmp)
        else:
          param = param.detach_()
          self.set_param(self, name, param)

  def set_param(self, curr_mod, name, param):
    if '.' in name:
      n = name.split('.')
      module_name = n[0]
      rest = '.'.join(n[1:])
      for name, mod in curr_mod.named_children():
        if module_name == name:
          self.set_param(mod, rest, param)
          break
    else:
      setattr(curr_mod, name, param)

  def detach_params(self):
    for name, param in self.named_params(self):
      self.set_param(self, name, param.detach())

  def copy(self, other, same_var=False):
    for name, param in other.named_params():
      if not same_var:
        param = to_var(param.data.clone(), requires_grad=True)
      self.set_param(name, param)


class MetaLinear(MetaModule):
  def __init__(self, *args, **kwargs):
    super().__init__()
    ignore = nn.Linear(*args, **kwargs)

    self.in_features = ignore.in_features
    self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
    if ignore.bias is not None:
      self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
    else:
      self.register_buffer('bias', None)

  def forward(self, x):
    return F.linear(x, self.weight, self.bias)

  def named_leaves(self):
    return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
  def __init__(self, *args, **kwargs):
    super().__init__()
    ignore = nn.Conv2d(*args, **kwargs)

    self.stride = ignore.stride
    self.padding = ignore.padding
    self.dilation = ignore.dilation
    self.groups = ignore.groups

    self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

    if ignore.bias is not None:
      self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
    else:
      self.register_buffer('bias', None)

  def forward(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

  def named_leaves(self):
    return [('weight', self.weight), ('bias', self.bias)]


# class MetaConvTranspose2d(MetaModule):
#   def __init__(self, *args, **kwargs):
#     super().__init__()
#     ignore = nn.ConvTranspose2d(*args, **kwargs)
#
#     self.stride = ignore.stride
#     self.padding = ignore.padding
#     self.dilation = ignore.dilation
#     self.groups = ignore.groups
#
#     self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
#
#     if ignore.bias is not None:
#       self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
#     else:
#       self.register_buffer('bias', None)
#
#   def forward(self, x, output_size=None):
#     output_padding = self._output_padding(x, output_size)
#     return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
#                               output_padding, self.groups, self.dilation)
#
#   def named_leaves(self):
#     return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
  def __init__(self, *args, **kwargs):
    super().__init__()
    ignore = nn.BatchNorm2d(*args, **kwargs)

    self.num_features = ignore.num_features
    self.eps = ignore.eps
    self.momentum = ignore.momentum
    self.affine = ignore.affine
    self.track_running_stats = ignore.track_running_stats

    if self.affine:
      self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
      self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(self.num_features))
      self.register_buffer('running_var', torch.ones(self.num_features))
    else:
      self.register_parameter('running_mean', None)
      self.register_parameter('running_var', None)

  def forward(self, x):
    return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)

  def named_leaves(self):
    return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm1d(MetaModule):
  def __init__(self, *args, **kwargs):
    super().__init__()
    ignore = nn.BatchNorm1d(*args, **kwargs)

    self.num_features = ignore.num_features
    self.eps = ignore.eps
    self.momentum = ignore.momentum
    self.affine = ignore.affine
    self.track_running_stats = ignore.track_running_stats

    if self.affine:
      self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
      self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(self.num_features))
      self.register_buffer('running_var', torch.ones(self.num_features))
    else:
      self.register_parameter('running_mean', None)
      self.register_parameter('running_var', None)

  def forward(self, x):
    return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)

  def named_leaves(self):
    return [('weight', self.weight), ('bias', self.bias)]


class meta_proj_head(MetaModule):
  def __init__(self, ch):
    super(meta_proj_head, self).__init__()
    self.in_features = ch

    self.fc1 = MetaLinear(ch, ch)
    self.bn1 = MetaBatchNorm1d(ch)
    self.fc2 = MetaLinear(ch, ch, bias=False)
    self.bn2 = MetaBatchNorm1d(ch)

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    # debug
    x = self.fc1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.fc2(x)
    x = self.bn2(x)

    return x