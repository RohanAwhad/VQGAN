# this is a temporary script to merge the preprocessed shards into a 5GB shard
import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

ROOT_DIR = '/scratch/rawhad/datasets/preprocessed_imagenet'
NEW_ROOT_DIR = '/scratch/rawhad/datasets/preprocessed_imagenet_5gb_shards'

# for train files
def merge(split):
  train_files = glob.glob(f'{ROOT_DIR}/{split}_*.pkl')
  new_train_shard = []
  shard_idx = 0
  for file in tqdm(train_files, desc=f'Merging {split} files'):
    with open(file, 'rb') as f:
      new_train_shard.append(pickle.load(f))

    if len(new_train_shard) > 5:
      new_train_shard = np.vstack(new_train_shard)
      with open(f'{NEW_ROOT_DIR}/{split}_{shard_idx}.pkl', 'wb') as f:
        pickle.dump(new_train_shard, f)

      shard_idx += 1
      new_train_shard = []

  if len(new_train_shard) > 0:
    new_train_shard = np.vstack(new_train_shard)
    with open(f'{NEW_ROOT_DIR}/{split}_{shard_idx}.pkl', 'wb') as f:
      pickle.dump(new_train_shard, f)


if __name__ == '__main__':
  os.makedirs(NEW_ROOT_DIR, exist_ok=True)
  merge('train')
  merge('test')
  print('done')
