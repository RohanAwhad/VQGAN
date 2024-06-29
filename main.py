from dotenv import load_dotenv
load_dotenv('/home/rawhad/personal_jobs/VQGAN/dev.env')

import argparse
import os
import torch

import engine, logger
from dataset import ImageNetDatasetLoaderLite, MNISTDatasetLoaderLite, CIFAR10DatasetLoaderLite
from new_model import Encoder, Generator, Discriminator, Codebook, VQGAN

# ===
# DDP setup
# ===
from torch.distributed import init_process_group, destroy_process_group

is_ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

# torchrun cmd sets env vars: RANK, LOCAL_RANK, and WORLD_SIZE
if is_ddp: 
  assert torch.cuda.is_available(), "DDP requires CUDA"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  print('DDP Rank:', ddp_rank)
  print('DDP Local Rank:', ddp_local_rank)
  print('DDP World Size:', ddp_world_size)
  DEVICE = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(DEVICE)
  is_master_process = ddp_rank == 0 # master process will do the logging, checkpointing, etc.
else:
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  DEVICE = 'cpu'
  if torch.cuda.is_available(): DEVICE = 'cuda'
  # if torch.backends.mps.is_available(): DEVICE = 'mps'  # seems to be a bug in MPS. Check W&B vqgan_hyperparam_search project run-2-test-cifar10-31-mps & -cuda
  is_master_process = True


# ===
# Constants
# ===
argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=3e-5)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--micro_batch_size', type=int, default=0)
argparser.add_argument('--n_steps', type=int, default=10000)
argparser.add_argument('--num_embeddings', type=int, default=4)
argparser.add_argument('--dropout_rate', type=float, default=0.1)
argparser.add_argument('--compression_factor', type=int, default=2)
argparser.add_argument('--codebook_dim', type=int, default=32)
argparser.add_argument('--disc_factor_threshold', type=float, default=2000)
argparser.add_argument('--data_dir', type=str, default="./data")
argparser.add_argument('--use_worker', action='store_true')
argparser.add_argument('--prefetch_size', type=int, default=1)
argparser.add_argument('--last_step', type=int, default=-1)
argparser.add_argument('--save_model', action='store_true')
argparser.add_argument('--do_overfit', action='store_true')
argparser.add_argument('--project_name', type=str, default='vqgan_hyperparam_search')
argparser.add_argument('--run_name', type=str, default='run-2-test-cifar10-6')
args = argparser.parse_args()

LR = args.lr
TOTAL_BATCH_SIZE = args.batch_size
MICRO_BATCH_SIZE = args.micro_batch_size if args.micro_batch_size > 0 else TOTAL_BATCH_SIZE
assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * ddp_world_size) == 0, "Total batch size must be divisible by (micro batch size * world_size)"

GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * ddp_world_size)
N_STEPS = args.n_steps
NUM_EMBEDDINGS = args.num_embeddings
CODEBOOK_EMBED_DIM = args.codebook_dim
DROPOUT_RATE = args.dropout_rate
COMPRESSION_FACTOR = args.compression_factor
USE_WORKER = args.use_worker
PREFETCH_SIZE = args.prefetch_size
DATA_DIR = args.data_dir
SAVE_MODEL = args.save_model
PROJECT_NAME = args.project_name
RUN_NAME = args.run_name
LAST_STEP = args.last_step
DO_OVERFIT = args.do_overfit
DISC_FACTOR_THRESHOLD = args.disc_factor_threshold

MAX_STEPS = min(N_STEPS // 3, 2408) if DO_OVERFIT else 2408 # ~1epoch
WARMUP_STEPS = int(MAX_STEPS * 0.037) # based on build_nanogpt ratio
MAX_LR = LR
MIN_LR = LR / 10

GEN_LR = LR
DISC_LR = LR

MODEL_DIR = '/scratch/rawhad/VQGAN/models'

if is_master_process:
  if SAVE_MODEL: os.makedirs(MODEL_DIR, exist_ok=True)
  LOGGER = logger.WandbLogger(project_name=PROJECT_NAME, run_name=RUN_NAME)
  LOGGER.update_config(vars(args))
  #LOGGER = logger.ConsoleLogger(project_name='vqgan', run_name='test-imagenet')
  print('GRAD_ACCUM_STEPS: ', GRAD_ACCUM_STEPS)
else:
  LOGGER = None

# ===
# Intialization
# ===
torch.manual_seed(1234)  # setting seed because we are using DDP
if torch.cuda.is_available(): torch.cuda.manual_seed(1234)

# train_ds = ImageNetDatasetLoaderLite(split='train', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR, process_rank=ddp_rank, world_size=ddp_world_size, prefetch_size=PREFETCH_SIZE, use_worker=USE_WORKER)
# if DO_OVERFIT: test_ds = train_ds
# else: test_ds = ImageNetDatasetLoaderLite(split='test', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR, process_rank=ddp_rank, world_size=ddp_world_size, prefetch_size=1)

# train_ds = MNISTDatasetLoaderLite(train=True, root=DATA_DIR, batch_size=MICRO_BATCH_SIZE, shuffle=True, download=False)
# test_ds = MNISTDatasetLoaderLite(train=False, root=DATA_DIR, batch_size=MICRO_BATCH_SIZE, shuffle=False, download=False)
train_ds = CIFAR10DatasetLoaderLite(train=True, root=DATA_DIR, batch_size=MICRO_BATCH_SIZE, shuffle=True, download=False)
test_ds = CIFAR10DatasetLoaderLite(train=True, root=DATA_DIR, batch_size=MICRO_BATCH_SIZE, shuffle=False, download=False)

IN_CHANNELS = 3

codebook = Codebook(num_embeddings=NUM_EMBEDDINGS, embedding_dim=CODEBOOK_EMBED_DIM)  # 2048 is the output dim of the encoder
encoder = Encoder(in_channels=IN_CHANNELS, out_channels=CODEBOOK_EMBED_DIM, m=COMPRESSION_FACTOR, dropout_rate=DROPOUT_RATE)
generator = Generator(in_channels=CODEBOOK_EMBED_DIM, out_channels=IN_CHANNELS, m=COMPRESSION_FACTOR, dropout_rate=DROPOUT_RATE)
discriminator = Discriminator(in_channels=IN_CHANNELS, num_filters_last=32, n_layers=2, dropout_rate=DROPOUT_RATE)
vqgan = VQGAN(encoder, codebook, generator)

# ===
# Configure Optimizers and LR Schedulers
# ===


# lr scheduler
lr_scheduler = engine.CosineLRScheduler(WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)

# Unable to compile
#vqgan = torch.compile(vqgan)
#discriminator = torch.compile(discriminator)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

training_config = engine.EngineConfig(
  train_ds=train_ds,
  test_ds=test_ds,
  vqgan=vqgan,
  disc=discriminator,
  N_STEPS=N_STEPS,
  disc_factor_threshold=DISC_FACTOR_THRESHOLD,
  device=DEVICE,
  logger=LOGGER,
  lr_scheduler=lr_scheduler,
  grad_accum_steps=GRAD_ACCUM_STEPS,
  checkpoint_every=1000,
  checkpoint_dir=MODEL_DIR,
  last_step=LAST_STEP,
  is_ddp=is_ddp,
  ddp_rank=ddp_rank,
  ddp_local_rank=ddp_local_rank,
  ddp_world_size=ddp_world_size,
  is_master_process=is_master_process,
  do_overfit=DO_OVERFIT,
  save_model=SAVE_MODEL,
)
#with torch.autograd.set_detect_anomaly(True):
engine.run(training_config)

if is_master_process:
  # save models
  codebook.to('cpu')
  encoder.to('cpu')
  generator.to('cpu')

  if SAVE_MODEL:
    torch.save(codebook.state_dict(), os.path.join(MODEL_DIR, 'codebook_final.pth'))
    torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder_final.pth'))
    torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'generator_final.pth'))


destroy_process_group()
