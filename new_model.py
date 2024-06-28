import torch
import torch.nn as nn
import torch.nn.functional as F

N_GROUPS = 64
START_CHANNEL = 64

class Upsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x: torch.Tensor):
    upsampled_x = F.interpolate(x, scale_factor=2, mode='nearest')
    return self.conv(upsampled_x)

class Downsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

  def forward(self, x: torch.Tensor):
    pad = (0, 1, 0, 1)
    downsampled_x = F.pad(x, pad, mode="constant", value=0)
    return self.conv(downsampled_x)

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.block = nn.Sequential(
      nn.GroupNorm(N_GROUPS, in_channels),
      nn.SiLU(),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.GroupNorm(N_GROUPS, out_channels),
      nn.SiLU(),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )
    if self.in_channels != self.out_channels:
      self.channel_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x: torch.Tensor):
    if self.in_channels != self.out_channels:
      return self.channel_up(x) + self.block(x)
    return x + self.block(x)


# ===
# Attention Block (Non-Local Block)
# X -> Norm -> Self-Attention (A) -> Projection Out -> X + A
#   |__________________________________________________^
# ===
class AttentionBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    
    self.in_channels = in_channels
    self.gn = nn.GroupNorm(32, in_channels)  # 32 is the default value in the paper
    self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x: torch.Tensor):
    batch_size, n_channels, n_rows, n_cols = x.shape
    h_ = self.gn(x)
    q = self.q(h_).view(batch_size, n_channels, -1).permute(0, 2, 1)
    k = self.k(h_).view(batch_size, n_channels, -1).permute(0, 2, 1)
    v = self.v(h_).view(batch_size, n_channels, -1).permute(0, 2, 1)

    y = F.scaled_dot_product_attention(q, k, v) # flash attention
    y = y.permute(0, 2, 1).contiguous().view(batch_size, n_channels, n_rows, n_cols)
    out = self.out(y)
    return x + out

# ===
# Codebook
# ===
class Codebook(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)  # QA (rohan): if we are detaching the q, down below, are we training these embeddings?
    self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    self.beta = 0.25

  def forward(self, x: torch.Tensor):
    batch_size, embed_dim, n_rows, n_cols = x.shape
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.flatten(0, 2).contiguous()


    a = torch.sum(x**2, dim=1, keepdim=True)
    b = torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0)
    ab2 = torch.matmul(x, self.embedding.weight.t())
    d = a + b - 2 * ab2
    indices = torch.argmin(d, dim=1)
    q = self.embedding(indices)

    # loss
    loss = torch.mean((x - q.detach())**2) + self.beta * torch.mean((x.detach() - q)**2)

    out: torch.Tensor = x + (q - x).detach()
    out = out.view(batch_size, n_rows, n_cols, embed_dim).permute(0, 3, 1, 2).contiguous()
    return out, indices, loss


# Number of layers and rough idea of the architecture is referenced from: https://github.com/dome272/VQGAN-pytorch
class Encoder(nn.Module):
  def __init__(self, in_channels, out_channels, m: int, dropout_rate: float):
    super().__init__()
    ch_mult = list(map(lambda x: 2**x, range(m)))
    self.start_channel = START_CHANNEL
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.m = m
    # Res -> Res -> Down
    layers = []
    layers.append(nn.Conv2d(in_channels, self.start_channel, kernel_size=3, stride=1, padding=1))
    in_channel = self.start_channel
    for ch_mul_factor in ch_mult:
      out_channel = self.start_channel * ch_mul_factor
      # layers.append(nn.Sequential(
      #   ResidualBlock(in_channel, out_channel),
      #   ResidualBlock(out_channel, out_channel),
      #   nn.Dropout2d(dropout_rate),
      # ))
      # in_channel = out_channel
      layers.append(nn.Sequential(
        ResidualBlock(in_channel, out_channel),
        ResidualBlock(out_channel, out_channel),
        Downsample(out_channel),
        nn.Dropout2d(dropout_rate),
      ))
      in_channel = out_channel

    layers.append(ResidualBlock(in_channel, in_channel))
    layers.append(AttentionBlock(in_channel))
    layers.append(ResidualBlock(in_channel, in_channel))
    layers.append(nn.GroupNorm(N_GROUPS, in_channel))
    layers.append(nn.SiLU())
    layers.append(nn.Dropout2d(dropout_rate))
    layers.append(nn.Conv2d(in_channel, out_channels, kernel_size=3, stride=1, padding=1))
    # layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0))  # pre-quant
    self.layers = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor):
    return self.layers(x)



class Generator(nn.Module):
  def __init__(self, in_channels, out_channels, m: int, dropout_rate: float):
    super().__init__()
    ch_mult = list(map(lambda x: 2**x, reversed(range(m))))
    self.start_channel = START_CHANNEL

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.m = m

    layers = []
    in_channel = self.start_channel * ch_mult[0]
    # layers.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0))  # post-quant
    layers.append(nn.Conv2d(self.in_channels, in_channel, kernel_size=3, stride=1, padding=1))
    layers.append(ResidualBlock(in_channel, in_channel))
    layers.append(AttentionBlock(in_channel))
    layers.append(ResidualBlock(in_channel, in_channel))
    layers.append(nn.Dropout2d(dropout_rate))

    # Res -> Res -> Up
    for ch_mul_factor in ch_mult:
      out_channel = self.start_channel * ch_mul_factor
      layers.append(nn.Sequential(
        ResidualBlock(in_channel, out_channel),
        ResidualBlock(out_channel, out_channel),
        ResidualBlock(out_channel, out_channel),
        Upsample(out_channel),
        nn.Dropout2d(dropout_rate),
      ))
      in_channel = out_channel
    
    layers.append(nn.GroupNorm(N_GROUPS, in_channel))
    layers.append(nn.SiLU())
    layers.append(nn.Dropout2d(dropout_rate))
    layers.append(nn.Conv2d(in_channel, self.out_channels, kernel_size=3, stride=1, padding=1))

    self.layers = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor):
    return self.layers(x)

"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""
class Discriminator(nn.Module):
  def __init__(self, in_channels, dropout_rate: float, num_filters_last=64, n_layers=3):
    super(Discriminator, self).__init__()

    layers = [nn.Conv2d(in_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
    num_filters_mult = 1

    for i in range(1, n_layers + 1):
      num_filters_mult_last = num_filters_mult
      num_filters_mult = min(2 ** i, 8)
      layers += [
        nn.Conv2d(
          in_channels=num_filters_last * num_filters_mult_last,
          out_channels=num_filters_last * num_filters_mult,
          kernel_size=4,
          stride=2 if i < n_layers else 1,
          padding=1,
          bias=False
        ),
        nn.BatchNorm2d(num_filters_last * num_filters_mult),
        nn.LeakyReLU(0.2, True),
        nn.Dropout2d(dropout_rate),
      ]

    layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)


# compiling the enc-codebook-gen models into a single model
class VQGAN(nn.Module):
  def __init__(self, encoder, codebook, generator):
    super().__init__()
    self.encoder = encoder
    self.codebook = codebook
    self.generator = generator

  def forward(self, inp):
    enc_inp = self.encoder(inp)
    tok_inp, indices, loss = self.codebook(enc_inp)
    fake_img = self.generator(tok_inp)

    return fake_img, indices, loss
