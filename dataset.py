import random
import torch
import torchvision

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
  def __init__(self, train: bool, root: str, batch_size: int, shuffle: bool):
    self.ds = torchvision.datasets.CIFAR10(root=root, train=train, download=False)
    self.batch_size = batch_size
    self.indices = list(range(len(self.ds)))
    if shuffle: random.shuffle(self.indices)

    self.to_tensor = ToTensor()
    self.reset()

  def reset(self):
    self.curr_idx = 0

  def next_batch(self):
    next_indices = self.indices[self.curr_idx:self.curr_idx+self.batch_size]
    self.curr_idx = (self.curr_idx + self.batch_size) % len(self.ds)
    
    imgs = []
    for idx in next_indices:
      img, _ = self.ds[idx]
      imgs.append(self.to_tensor(img).flatten())

    return torch.stack(imgs)
