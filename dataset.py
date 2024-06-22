import glob
import os
import pickle
import random
import torch

from abc import ABC, abstractmethod

class Dataset(ABC):
  @abstractmethod
  def next_batch(self):
    pass

  @abstractmethod
  def reset(self):
    pass

class ImageNetDatasetLoaderLite(Dataset):
  def __init__(self, root: str, split: str, batch_size: int, process_rank: int, num_processes: int):
    self.root = root
    self.batch_size = batch_size
    self.split = split
    self.process_rank = process_rank
    self.num_processes = num_processes

    self.files = glob.glob(os.path.join(root, f'{split}_*.pkl'))
    self.curr_file_ptr = None
    self.reset()

  def reset(self):
    if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
      self.curr_file_ptr = 0
      self.load_shard()
    self.curr_idx = self.batch_size * self.process_rank

  def load_shard(self):
    with open(self.files[self.curr_file_ptr], 'rb') as f:
      self.curr_shard = pickle.load(f)

  def next_batch(self):
    batch = torch.from_numpy(self.curr_shard[self.curr_idx:self.curr_idx+self.batch_size])
    self.curr_idx += (self.batch_size * self.num_processes)
    # drop last batch if it's smaller than batch_size
    if (self.curr_idx + (self.batch_size * self.num_processes)) >= len(self.curr_shard):
      self.curr_file_ptr = (self.curr_file_ptr + 1) % len(self.files)
      self.curr_idx = self.batch_size * self.process_rank
      self.load_shard()
    
    # dropping last batch so commenting this out
    # if len(batch) < self.batch_size:
    #   remainder = self.batch_size - len(batch)
    #   batch = torch.vstack((batch, torch.from_numpy(self.curr_shard[:remainder])))
    #   self.curr_idx = remainder
    
    return {'images': batch}



class MNISTDatasetLoaderLite(Dataset):
  def __init__(self, train: bool, root: str, batch_size: int, shuffle: bool):
    import torchvision
    from torchvision.transforms import ToTensor

    self.ds = torchvision.datasets.MNIST(root=root, train=train, download=True)
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
      img_tensor = self.to_tensor(img)
      imgs.append(img_tensor)

    return {
      'images': torch.stack(imgs),
    }
