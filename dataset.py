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
  def __init__(self, train: bool, root: str, batch_size: int, shuffle: bool, patch_size: int):
    self.ds = torchvision.datasets.CIFAR10(root=root, train=train, download=False)
    self.batch_size = batch_size
    self.indices = list(range(len(self.ds)))
    if shuffle: random.shuffle(self.indices)

    self.to_tensor = ToTensor()
    self.patch_size = patch_size
    self.reset()

  def reset(self):
    self.curr_idx = 0

  def next_batch(self):
    next_indices = self.indices[self.curr_idx:self.curr_idx+self.batch_size]
    self.curr_idx = (self.curr_idx + self.batch_size) % len(self.ds)
    
    imgs = []
    n_rows_list, n_cols_list = [], []
    for idx in next_indices:
      img, _ = self.ds[idx]
      img_tensor = self.to_tensor(img)
      n_channels, n_rows, n_cols = img_tensor.size()
      img_tensor = img_tensor.view(n_channels, n_rows//self.patch_size, self.patch_size, n_cols//self.patch_size, self.patch_size)
      img_tensor = img_tensor.permute(1, 3, 0, 2, 4).flatten(2)
      n_rows_list.append(img_tensor.size(0))
      n_cols_list.append(img_tensor.size(1))
      img_tensor = img_tensor.flatten(0, 1)  # (n_patches, patch_size*patch_size*n_channels)
      imgs.append(img_tensor)

    # assert all values in n_rows_list and n_cols_list are the same
    assert all([n_rows_list[0] == n_rows for n_rows in n_rows_list])
    assert all([n_cols_list[0] == n_cols for n_cols in n_cols_list])

    return {
      'imgs': torch.stack(imgs),
      'n_rows': n_rows_list[0],
      'n_cols': n_cols_list[0],
    }
