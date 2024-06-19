import os
import pickle
import random
import torch

from datasets import load_dataset
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm


# ===
# Constants
# ===
SAVE_DIR = "/scratch/rawhad/datasets/preprocessed_imagenet"
os.makedirs(SAVE_DIR, exist_ok=True)

TRANSFORM = v2.Compose([
  v2.PILToTensor(),
  v2.Resize(224)
])

SHARD_SIZE = 2000

# ===
# Utils
# ===
def save_pickle(obj, path):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)


# ===
# Main
# ===
train_ds = load_dataset('ILSVRC/imagenet-1k', split='train')
test_ds = load_dataset('ILSVRC/imagenet-1k', split='test')

train_indices = list(range(len(train_ds)))
random.shuffle(train_indices)
test_indices = list(range(len(test_ds)))


def process_ds(indices, ds, split):
  imgs = []
  shard_id = 0
  for idx in tqdm(indices):
    img: Image.Image = ds[idx]['image']
    w, h = img.size
    new_w = min(w, h)
    # center crop
    img = v2.functional.center_crop(img, (new_w, new_w))
    img_tensor = TRANSFORM(img) / 255.0
    imgs.append(img_tensor)

    if len(imgs) >= SHARD_SIZE:
      # save the images
      imgs = torch.stack(imgs).numpy()
      save_pickle(imgs, os.path.join(SAVE_DIR, f'{split}_{shard_id:4d}.pkl'))

      # reset
      imgs = []
      shard_id += 1

  if len(imgs) > 0:
    # save the images
    imgs = torch.stack(imgs).numpy()
    save_pickle(imgs, os.path.join(SAVE_DIR, f'{split}_{shard_id:4d}.pkl'))


process_ds(train_indices, train_ds, 'train')
process_ds(test_indices, test_ds, 'test')