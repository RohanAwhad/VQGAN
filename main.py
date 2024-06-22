from dotenv import load_dotenv
load_dotenv('/home/rawhad/personal_jobs/VQGAN/dev.env')

import argparse
import os
import torch

import engine, logger
from dataset import ImageNetDatasetLoaderLite
from model import Encoder, Generator, Discriminator, Codebook, VQGAN

# ===
# Constants
# ===
argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--n_steps', type=int, default=5000)
argparser.add_argument('--num_embeddings', type=int, default=1024)
argparser.add_argument('--dropout_rate', type=float, default=0.1)
argparser.add_argument('--data_dir', type=str, default="/scratch/rawhad/datasets/preprocessed_tiny_imagenet")
argparser.add_argument('--last_step', type=int, default=-1)
argparser.add_argument('--save_model', action='store_true')
argparser.add_argument('--project_name', type=str, default='vqgan')
argparser.add_argument('--run_name', type=str, default='test-imagenet')
args = argparser.parse_args()

LR = args.lr
TOTAL_BATCH_SIZE = args.batch_size
MICRO_BATCH_SIZE = args.micro_batch_size
assert TOTAL_BATCH_SIZE % MICRO_BATCH_SIZE == 0, "Total batch size must be divisible by micro batch size"
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE
N_STEPS = args.n_steps
NUM_EMBEDDINGS = args.num_embeddings
DROPOUT_RATE = args.dropout_rate
DATA_DIR = args.data_dir
SAVE_MODEL = args.save_model
PROJECT_NAME = args.project_name
RUN_NAME = args.run_name

WARMUP_STEPS = 5000
MAX_STEPS = N_STEPS // 3
MAX_LR = LR
MIN_LR = LR / 10

GEN_LR = LR
DISC_LR = LR

DEVICE = 'cpu'
if torch.cuda.is_available(): DEVICE = 'cuda'
if torch.backends.mps.is_available(): DEVICE = 'mps'

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

LOGGER = logger.WandbLogger(project_name=PROJECT_NAME, run_name=RUN_NAME)
#LOGGER = logger.ConsoleLogger(project_name='vqgan', run_name='test-imagenet')

# ===
# Intialization
# ===
train_ds = ImageNetDatasetLoaderLite(split='train', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR)
test_ds = ImageNetDatasetLoaderLite(split='train', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR)

codebook = Codebook(num_embeddings=NUM_EMBEDDINGS, embedding_dim=2048)  # 2048 is the output dim of the encoder
encoder = Encoder(DROPOUT_RATE)
generator = Generator(DROPOUT_RATE)
discriminator = Discriminator(DROPOUT_RATE)
vqgan = VQGAN(encoder, codebook, generator)


vqgan_opt = torch.optim.AdamW(vqgan.parameters(), lr=GEN_LR)
disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=DISC_LR)

# Unable to compile
#vqgan = torch.compile(vqgan)
#discriminator = torch.compile(discriminator)
torch.set_float32_matmul_precision('high')

# lr scheduler
lr_scheduler = engine.CosineLRScheduler(WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
training_config = engine.EngineConfig(
  train_ds=train_ds,
  test_ds=test_ds,
  vqgan=vqgan,
  discriminator=discriminator,
  vqgan_opt=vqgan_opt,
  disc_opt=disc_opt,
  N_STEPS=N_STEPS,
  device=DEVICE,
  logger=LOGGER,
  lr_scheduler=lr_scheduler,
  grad_accum_steps=GRAD_ACCUM_STEPS,
  checkpoint_every=1000,
  checkpoint_dir=MODEL_DIR,
  last_step=args.last_step,
)
engine.run(training_config)

# save models
codebook.to('cpu')
encoder.to('cpu')
generator.to('cpu')

if SAVE_MODEL:
  torch.save(codebook.state_dict(), os.path.join(MODEL_DIR, 'codebook_final.pth'))
  torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder_final.pth'))
  torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'generator_final.pth'))
