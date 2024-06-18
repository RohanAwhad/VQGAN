import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torchvision.transforms import ToTensor

class Dataset:
  @abstractmethod
  def next_batch(self):
    pass

  @abstractmethod
  def reset(self):
    pass


class DatasetLoaderLite(Dataset):
  def __init__(self, train: bool, root: str, batch_size: int, shuffle: bool, patch_size: int):
    self.ds = torchvision.datasets.CIFAR10(root=root, train=train, download=False)
    self.batch_size = batch_size
    self.indices = list(range(len(self.ds)))
    if shuffle: random.shuffle(self.indices)

    self.to_tensor = ToTensor()
    self.pad = nn.ZeroPad2d(2)  # to convert 28x28 MNIST to 32x32
    self.patch_size = patch_size
    self.reset()

  def reset(self):
    self.curr_idx = 0

  def next_batch(self):
    next_indices = self.indices[self.curr_idx:self.curr_idx+self.batch_size]
    self.curr_idx = (self.curr_idx + self.batch_size) % len(self.ds)
    
    imgs = []
    for idx in next_indices:
      img, _ = self.ds[idx]
      img_tensor = self.to_tensor(img)
      imgs.append(img_tensor)

    return {
      'images': torch.stack(imgs),
    }
