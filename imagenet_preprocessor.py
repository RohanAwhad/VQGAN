import pickle
import random
import torch

from datasets import load_dataset
from torchvision.transforms import v2

def save_pickle(obj, path):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

TRANSFORM = v2.Compose([
  v2.PILToTensor(),
  v2.Resize(224)
])

train_ds = load_dataset('ILSVRC/imagenet-1k', split='train')
test_ds = load_dataset('ILSVRC/imagenet-1k', split='test')

random.shuffle(train_ds)

# do train_ds
SHARD_SIZE = 2000
SHARD_CNT = 0
imgs = []
for x in train_ds:
  img = x['image']
  w, h = img.size
  new_w = min(w, h)
  # center crop
  img = v2.functional.center_crop(img, (new_w, new_w))
  img_tensor = TRANSFORM(img) / 255.0
  imgs.append(img_tensor)

  if len(imgs) >= SHARD_SIZE:
    # save the images
    imgs = torch.stack(imgs).numpy()
    save_pickle(imgs, f'train_imgs_{SHARD_CNT:4d}.pkl')

    # reset
    imgs = []
    SHARD_CNT += 1

if len(imgs) > 0:
  # save the images
  imgs = torch.stack(imgs).numpy()
  save_pickle(imgs, f'train_imgs_{SHARD_CNT:4d}.pkl')