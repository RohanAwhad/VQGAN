import glob
import os
import pickle
import random
import threading
import time
import torch

from collections import deque
from abc import ABC, abstractmethod

class Dataset(ABC):
  @abstractmethod
  def next_batch(self):
    pass

  @abstractmethod
  def reset(self):
    pass

class ImageNetDatasetLoaderLite(Dataset):
  def __init__(self, root: str, split: str, batch_size: int, process_rank: int, num_processes: int, prefetch_size: int = 1):
    self.root = root
    self.batch_size = batch_size
    self.split = split
    self.process_rank = process_rank
    self.num_processes = num_processes

    self.files = glob.glob(os.path.join(root, f'{split}_*.pkl'))
    self.curr_file_ptr = None
    self.prefetch_size = prefetch_size
    self.prefetch_thread = None
    self.reset()

  def reset(self):
    if self.curr_file_ptr is None or self.curr_file_ptr != 0:  # loading shard is costly, and this op is common during overfitting or hyperparameter tuning
      self.curr_file_ptr = 0
      self.load_shard()
    self.curr_idx = self.batch_size * self.process_rank

    # for efficiency reasons, we build a queue to prefetch batches
    self.prefetch_queue = deque(maxlen=self.prefetch_size)
    if self.prefetch_thread is None:
      self.prefetch_thread = threading.Thread(target=self._fill_queue)
      self.prefetch_thread.start()

  def _fill_queue(self):
    while len(self.prefetch_queue) < self.prefetch_size:
      # add a batch to the queue
      self.prefetch_queue.append(self._next_batch())
      time.sleep(0.25)

  def load_shard(self):
    with open(self.files[self.curr_file_ptr], 'rb') as f:
      self.curr_shard = pickle.load(f)

  def _next_batch(self):
    batch = torch.from_numpy(self.curr_shard[self.curr_idx:self.curr_idx+self.batch_size])
    self.curr_idx += (self.batch_size * self.num_processes)
    # drop last batch if it's smaller than batch_size
    if (self.curr_idx + (self.batch_size * self.num_processes)) >= len(self.curr_shard):
      self.curr_file_ptr = (self.curr_file_ptr + 1) % len(self.files)
      self.curr_idx = self.batch_size * self.process_rank
      self.load_shard()
    
    return {'images': batch}

  def next_batch(self):
    if len(self.prefetch_queue) == 0: return self._next_batch()
    return self.prefetch_queue.popleft()



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
