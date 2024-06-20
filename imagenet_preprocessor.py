import multiprocessing as mp
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

def process(img: Image.Image):
  img = img.convert('RGB')
  w, h = img.size
  new_w = min(w, h)
  # center crop
  img = v2.functional.center_crop(img, (new_w, new_w))
  img_tensor = TRANSFORM(img) / 255.0


def process_ds(indices, ds, split):
  n_procs = os.cpu_count() - 1
  with mp.Pool(n_procs) as pool:
    imgs = []
    shard_id = 0
    pbar = tqdm(total=len(indices), desc=f'Processing {split}')
    for img_tensor in pool.imap_unordered(process, map(lambda idx: ds[idx], indices), chunksize=50):
      pbar.update(1)
      imgs.append(img_tensor)

      if len(imgs) >= SHARD_SIZE:
        # save the images
        imgs = torch.stack(imgs).numpy()
        save_pickle(imgs, os.path.join(SAVE_DIR, f'{split}_{shard_id}.pkl'))

        # reset
        imgs = []
        shard_id += 1

    if len(imgs) > 0:
      # save the images
      imgs = torch.stack(imgs).numpy()
      save_pickle(imgs, os.path.join(SAVE_DIR, f'{split}_{shard_id}.pkl'))



if __name__ == '__main__':
  process_ds(train_indices, train_ds, 'train')
  process_ds(test_indices, test_ds, 'test')