import random
import glob
import torch

from datasets import load_dataset
from PIL import Image
from torchvision.transforms import v2

TRANSFORM = v2.Compose([
  v2.PILToTensor(),
  v2.CenterCrop(224),
  v2.Resize(224)
])

class DatasetLoaderLite:
  def __init__(self, split: str, batch_size: int, shuffle: bool):
    self.ds = load_dataset('ILSVRC/imagenet-1k', split=split)
    self.batch_size = batch_size
    self.indices = list(range(len(self.ds)))
    if shuffle: random.shuffle(self.indices)

    self.transform = TRANSFORM
    self.reset()

  def reset(self):
    self.curr_idx = 0

  def next_batch(self):
    next_indices = self.indices[self.curr_idx:self.curr_idx+self.batch_size]
    self.curr_idx = (self.curr_idx + self.batch_size) % len(self.ds)
    
    imgs = []
    for idx in next_indices:
      img = self.ds[idx]['image'].convert('RGB')
      img_tensor = self.transform(img) / 255.0
      imgs.append(img_tensor)

    return {'images': torch.stack(imgs)}
