import os
import pickle
import random
import torch

from datasets import load_dataset
from torchvision.transforms import v2
from tqdm import tqdm

def save_pickle(obj, path):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

TRANSFORM = v2.Compose([
  v2.PILToTensor(),
  v2.Resize(224)
])

train_ds = load_dataset('ILSVRC/imagenet-1k', split='train')
test_ds = load_dataset('ILSVRC/imagenet-1k', split='test')

train_indices = list(range(len(train_ds)))
random.shuffle(train_indices)

# do train_ds
SAVE_DIR = "/scratch/rawhad/datasets/preprocessed_imagenet"
os.makedirs(SAVE_DIR, exist_ok=True)

SHARD_SIZE = 2000
SHARD_CNT = 0
imgs = []
for idx in tqdm(train_indices):
  img = train_ds[idx]['image']
  w, h = img.size
  new_w = min(w, h)
  # center crop
  img = v2.functional.center_crop(img, (new_w, new_w))
  img_tensor = TRANSFORM(img) / 255.0
  imgs.append(img_tensor)

  if len(imgs) >= SHARD_SIZE:
    # save the images
    imgs = torch.stack(imgs).numpy()
    save_pickle(imgs, os.path.join(SAVE_DIR, f'train_imgs_{SHARD_CNT:4d}.pkl'))

    # reset
    imgs = []
    SHARD_CNT += 1

if len(imgs) > 0:
  # save the images
  imgs = torch.stack(imgs).numpy()
  save_pickle(imgs, os.path.join(SAVE_DIR, f'train_imgs_{SHARD_CNT:4d}.pkl'))
