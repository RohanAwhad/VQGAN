import glob
import multiprocessing as mp
import os
import pickle
import random
import threading
import time
import torch

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

class Dataset(ABC):
  @abstractmethod
  def next_batch(self):
    pass

  @abstractmethod
  def reset(self):
    pass

class ImageNetDatasetLoaderLite(Dataset):
  def __init__(self, root: str, split: str, batch_size: int, process_rank: int, world_size: int, use_worker: bool = False, prefetch_size: int = 1):
    self.ds_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    self.ds_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    self.root = root
    self.batch_size = batch_size
    self.split = split
    self.process_rank = process_rank
    self.world_size = world_size

    self.files = glob.glob(os.path.join(root, f'{split}_*.pkl'))
    self.shard_size = self._get_shard_size()
    self.curr_file_ptr = None
    
    # multiprocessing
    self.use_worker = use_worker
    self.prefetch_size = prefetch_size
    self.workers: list[mp.Process] = []
    self.prefetch_thread = None

    self.offset = self.batch_size * self.process_rank
    self.step = self.batch_size * self.world_size
    assert self.step <= self.shard_size, "Batch size * world size must be less than or equal to shard size"

    self.reset()

  def next_batch(self):
    if self.use_worker: return self._next_batch_from_queue()
    return self._next_batch()

  def _get_shard_size(self): return self.load_shard(0).shape[0]

  def load_shard(self, file_ptr):
    with open(self.files[file_ptr], 'rb') as f:
      return pickle.load(f)

  def _next_batch(self):
    batch = torch.from_numpy(self.curr_shard[self.curr_idx:self.curr_idx+self.batch_size])
    self.curr_idx += self.step
    # drop last batch if it's smaller than batch_size
    if (self.curr_idx + self.step) >= len(self.curr_shard):
      self.curr_file_ptr = (self.curr_file_ptr + 1) % len(self.files)  # cycle through files if necessary
      self.curr_idx = self.offset
      self.curr_shard = self.load_shard(self.curr_file_ptr)

    # normalize
    batch = (batch - self.ds_mean) / self.ds_std
    
    return {'images': batch}

  def reset(self):
    if not self.use_worker:
      if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
        self.curr_file_ptr = 0
        self.curr_shard = self.load_shard(self.curr_file_ptr)
      self.curr_idx = self.offset

    else:
      # for efficiency reasons, we build a queue to prefetch batches
      self.prefetch_queue = mp.Queue(maxsize=self.prefetch_size)
      if len(self.workers) > 0:
        for worker in self.workers:
          worker.terminate()
          worker.close()
      self.workers = []
      worker = mp.Process(target=self.worker)
      worker.start()
      self.workers.append(worker)


  def worker(self):
    if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
      self.curr_file_ptr = 0
      self.curr_shard = self.load_shard(self.curr_file_ptr)
    self.curr_idx = self.offset
    self._fill_queue()

  def _fill_queue(self):
    while True:
      # add a batch to the queue
      # multiprocessing
      if self.prefetch_queue.full(): time.sleep(0.25)
      else: self.prefetch_queue.put(self._next_batch())

  def _next_batch_from_queue(self):
    # multiprocessing
    while self.prefetch_queue.empty():
      time.sleep(0.25)
    return self.prefetch_queue.get()



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
