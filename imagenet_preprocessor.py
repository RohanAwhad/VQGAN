import glob
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

SHARD_SIZE = 2000 * 5 # 2000 img ~1GB so 5GB per shard

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

TRAIN_INDICES_PATH = 'shuffled_train_indices.txt'
if os.path.exists(TRAIN_INDICES_PATH):
  with open(TRAIN_INDICES_PATH, 'r') as f:
    train_indices = [int(x) for x in f.read().split('\n') if x]
else:
  train_indices = list(range(len(train_ds)))
  random.shuffle(train_indices)
  with open(TRAIN_INDICES_PATH, 'w') as f:
    f.write('\n'.join([str(x) for x in train_indices]))

test_indices = list(range(len(test_ds)))

def process(img: Image.Image):
  img = img.convert('RGB')
  w, h = img.size
  new_w = min(w, h)
  # center crop
  img = v2.functional.center_crop(img, (new_w, new_w))
  img_tensor = TRANSFORM(img) / 255.0
  return img_tensor

def restore(split):
  files = glob.glob(f'{SAVE_DIR}/{split}_*.pkl')
  return len(files)

def process_ds(indices, ds, split):
  n_procs = max(1, os.cpu_count()//2)
  print(f'Using {n_procs} processes')
  with mp.Pool(n_procs) as pool:
    imgs = []
    shard_id = 0
    pbar = tqdm(total=len(indices), desc=f'Processing {split}')
    # restore
    shard_id = restore(split)
    n_processed = shard_id * SHARD_SIZE
    indices = indices[n_processed:]
    pbar.update(n_processed)


    for img_tensor in pool.imap_unordered(process, map(lambda idx: ds[idx]['image'], indices), chunksize=100):
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
