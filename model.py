import torch
import torch.nn as nn
import torch.nn.functional as F

# upsample
class UpSample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3)

  def forward(self, x):
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    return self.conv(x)

# downsample
class DownSample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
  def forward(self, x):
    return self.conv(x)


# Encoder
class Encoder(nn.Module):
  def __init__(self, input_dim: int, output_dim: int):
    super(Encoder, self).__init__()
    dropout_rate = 0.1
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(256, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(128, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(64, 32),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(32, 16),
      nn.BatchNorm1d(16),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(16, 8),
      nn.BatchNorm1d(8),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(8, output_dim)
    )

  def forward(self, x):
    return self.layers(x)


# Generator
class Generator(nn.Module):
  def __init__(self, latent_dim: int, output_dim: int):
    super(Generator, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(latent_dim, 128),
      nn.BatchNorm1d(128),
      nn.GELU(),
      nn.Linear(128, 256),
      nn.BatchNorm1d(256),
      nn.GELU(),
      nn.Linear(256, 512),
      nn.BatchNorm1d(512),
      nn.GELU(),
      nn.Linear(512, 1024),
      nn.BatchNorm1d(1024),
      nn.GELU(),
      nn.Linear(1024, 2048),
      nn.BatchNorm1d(2048),
      nn.GELU(),
      nn.Linear(2048, 2048),
      nn.BatchNorm1d(2048),
      nn.GELU(),
      nn.Linear(2048, 1024),
      nn.BatchNorm1d(1024),
      nn.GELU(),
      nn.Linear(1024, 512),
      nn.BatchNorm1d(512),
      nn.GELU(),
      nn.Linear(512, output_dim),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.layers(x)


# Discriminator
class Discriminator(nn.Module):
  def __init__(self, input_dim: int):
    super(Discriminator, self).__init__()
    dropout_rate = 0.3
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(128, 1),
    )

  def forward(self, x):
    return self.layers(x)


# codebook
class Codebook(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super(Codebook, self).__init__()
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim

  def forward(self, x):
    batch_size = x.shape[0]
    # calculate euclidean distance between x and each embedding
    dist = torch.norm(x.unsqueeze(1) - self.embedding.weight, dim=2)
    indices = torch.argmin(dist, dim=1).unsqueeze(1)  # shape = (batch_size, 1)
    # create ohe vector with indices == 1 and other == 0
    ohe = torch.zeros((batch_size, self.num_embeddings), device=x.device)
    ohe.scatter_(1, indices, 1)
    ohe = ohe.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
    out = (ohe * self.embedding.weight).sum(dim=1)
    # to let the grads flow
    out = x + out - x.detach()
    return out


# compiling the enc-codebook-gen models into a single model
class VQGAN(nn.Module):
  def __init__(self, encoder, codebook, generator):
    super(VQGAN, self).__init__()
    self.encoder = encoder
    self.codebook = codebook
    self.generator = generator

  def forward(self, inp):
    enc_inp = self.encoder(inp)
    tok_inp = self.codebook(enc_inp)
    fake_img = self.generator(tok_inp)

    # loss calculation
    sg_enc_inp = enc_inp.detach()
    sg_tok_inp = tok_inp.detach()
    loss = torch.norm(sg_enc_inp - tok_inp) + torch.norm(sg_tok_inp - enc_inp)
    return fake_img, loss