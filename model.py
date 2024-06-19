import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===
# Encoder
# ===
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, apply_act=True):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.apply_act = apply_act
    if self.apply_act: self.act = nn.ReLU()

  def forward(self, x):
    x = self.bn(self.conv(x))
    if self.apply_act: x = self.act(x)
    return x

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample=False):
    super().__init__()

    self.stride = 2 if downsample else 1
    self.conv_blocks = nn.Sequential(
      ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
      ConvBlock(out_channels, out_channels, kernel_size=3, stride=self.stride, padding=1),
      ConvBlock(out_channels, 4*out_channels, kernel_size=1, stride=1, padding=0, apply_act=False),  # expansion block,
    )
    self.shortcut = ConvBlock(in_channels, 4*out_channels, kernel_size=1, stride=self.stride, padding=0, apply_act=False)
    self.act = nn.ReLU()

  def forward(self, x):
    conv_out = self.conv_blocks(x)
    shortcut_out = self.shortcut(x)
    x = self.act(conv_out + shortcut_out)
    return x

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    dropout_rate = 0.2

    self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Sequential(
      ResBlock(in_channels=64, out_channels=64),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=256, out_channels=64),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=256, out_channels=64),
    )
    self.conv3 = nn.Sequential(
      ResBlock(in_channels=256, out_channels=128, downsample=True),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=512, out_channels=128),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=512, out_channels=128),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=512, out_channels=128),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=512, out_channels=128),
    )
    self.conv4 = nn.Sequential(
      ResBlock(in_channels=512, out_channels=256, downsample=True),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=1024, out_channels=256),
    )
    self.conv5 = nn.Sequential(
      ResBlock(in_channels=1024, out_channels=512, downsample=True),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=2048, out_channels=512),
      nn.Dropout2d(p=dropout_rate),
      ResBlock(in_channels=2048, out_channels=512),
    )



  def forward(self, x):
    x = self.conv1(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    return x

# ===
# Generator model
# ===
class DeconvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, apply_act=True):
    super().__init__()
    self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.apply_act = apply_act
    if self.apply_act:
        self.act = nn.ReLU()

  def forward(self, x):
    x = self.bn(self.deconv(x))
    if self.apply_act: x = self.act(x)
    return x


class UpResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, upsample=False):
    super().__init__()

    self.stride = 2 if upsample else 1
    self.deconv_blocks = nn.Sequential(
      DeconvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0),
      DeconvBlock(out_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, output_padding=1 if upsample else 0),
      DeconvBlock(out_channels, out_channels*4, kernel_size=1, stride=1, padding=0, output_padding=0, apply_act=False),  # contraction block
    )
    self.shortcut = DeconvBlock(in_channels, out_channels*4, kernel_size=1, stride=self.stride, padding=0, output_padding=1 if upsample else 0, apply_act=False)
    self.act = nn.ReLU()

  def forward(self, x):
    deconv_out = self.deconv_blocks(x)
    shortcut_out = self.shortcut(x)
    x = self.act(deconv_out + shortcut_out)
    return x


class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    dropout_rate = 0.2
    self.deconv5 = nn.Sequential(
      UpResBlock(in_channels=2048, out_channels=512),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=2048, out_channels=512),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=2048, out_channels=256, upsample=True),
    )

    self.deconv4 = nn.Sequential(
      UpResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=1024, out_channels=256),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=1024, out_channels=128, upsample=True),
    )

    self.deconv3 = nn.Sequential(
      UpResBlock(in_channels=512, out_channels=128),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=512, out_channels=128),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=512, out_channels=64, upsample=True),
      nn.Dropout2d(p=dropout_rate),
    )
    self.deconv2 = nn.Sequential(
      UpResBlock(in_channels=256, out_channels=64),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=256, out_channels=64),
      nn.Dropout2d(p=dropout_rate),
      UpResBlock(in_channels=256, out_channels=16, upsample=True),
    )
    self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1)
    self.act = nn.Sigmoid()

  def forward(self, x):
    x = self.deconv5(x)
    x = self.deconv4(x)
    x = self.deconv3(x)
    x = self.deconv2(x)
    x = self.deconv1(x)
    x = self.act(x)
    return x

# ===
# Discriminator model
# ===
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    dropout_rate = 0.2
    self.encoder = Encoder()
    self.dropout = nn.Dropout2d(p=dropout_rate)
    self.fc = nn.Linear(2048, 1)
  
  def forward(self, x):
    x = self.encoder(x)
    x = self.dropout(x)
    x = self.fc(x.permute(0, 2, 3, 1)).sigmoid()
    return x


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

  def forward(self, x):
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
    loss = torch.mean((x.detach() - q)**2) + torch.mean((x - q.detach())**2)

    out = x + (q - x).detach()
    out = out.view(batch_size, n_rows, n_cols, embed_dim).permute(0, 3, 1, 2).contiguous()
    return out, indices, loss


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
