import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclasses.dataclass
class EncoderConfig:
  input_dim: int
  output_dim: int
  max_len: int
  n_hidden_layers: int
  n_hidden_dims: list
  merge_after: int
  intermediate_scale: int
  n_heads: int
  dropout_rate: float

@dataclasses.dataclass
class GeneratorConfig:
  input_dim: int
  output_dim: int
  max_len: int
  n_hidden_layers: int
  n_hidden_dims: list
  merge_after: int
  intermediate_scale: int
  n_heads: int
  dropout_rate: float

@dataclasses.dataclass
class DiscriminatorConfig:
  input_dim: int
  max_len: int
  n_hidden_layers: int
  embed_dim: int
  intermediate_scale: int
  n_heads: int
  dropout_rate: float

@dataclasses.dataclass
class CodebookConfig:
  num_embeddings: int
  embedding_dim: int

# TODO (rohan): add dropouts

class AttentionBlock(nn.Module):
  def __init__(self, embed_dim, n_heads):
    super().__init__()
    assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
    self.embed_dim = embed_dim
    self.n_heads = n_heads
    self.head_dim = embed_dim // n_heads
    self.q = nn.Linear(embed_dim, embed_dim, bias=True)
    self.k = nn.Linear(embed_dim, embed_dim, bias=True)
    self.v = nn.Linear(embed_dim, embed_dim, bias=True)
    self.out = nn.Linear(embed_dim, embed_dim, bias=True)

  def forward(self, x):
    batch_size = x.size(0)
    q = self.q(x).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    k = self.k(x).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    v = self.v(x).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    attn = torch.einsum('bhid,bhjd->bhij', q, k) / self.head_dim**0.5
    attn = F.softmax(attn, dim=-1)
    x = torch.einsum('bhij,bhjd->bhid', attn, v).permute(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
    x = self.out(x)
    return x

class MLP(nn.Module):
  def __init__(self, embed_dim, intermediate_dim):
    super().__init__()
    self.fc1 = nn.Linear(embed_dim, intermediate_dim)
    self.fc2 = nn.Linear(intermediate_dim, embed_dim)

  def forward(self, x):
    return self.fc2(F.gelu(self.fc1(x)))

class EncoderBlock(nn.Module):
  def __init__(self, embed_dim, intermediate_dim, n_heads, dropout_rate):
    super().__init__()
    self.attn_block = AttentionBlock(embed_dim, n_heads)
    self.mlp = MLP(embed_dim, intermediate_dim)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):
    x = x + self.dropout(self.attn_block(self.ln1(x)))
    x = x + self.dropout(self.mlp(self.ln2(x)))
    return x

class PatchMerge(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, n_rows, n_cols):
    batch_size, n_patches, embed_dim = x.size()
    x = x.view(batch_size, n_rows, n_cols, embed_dim)
    tl = x[:, ::2, ::2, :].unsqueeze(-2)
    tr = x[:, ::2, 1::2, :].unsqueeze(-2)
    bl = x[:, 1::2, ::2, :].unsqueeze(-2)
    br = x[:, 1::2, 1::2, :].unsqueeze(-2)
    x = torch.cat([tl, tr, bl, br], dim=-2).mean(dim=-2).flatten(1, 2)
    return x, n_rows//2, n_cols//2



class Encoder(nn.Module):
  def __init__(self, input_dim, output_dim, max_len, n_hidden_layers, n_hidden_dims, merge_after, intermediate_scale, n_heads, dropout_rate):
    super().__init__()

    self.patch_embedder = nn.Linear(input_dim, n_hidden_dims[0], bias=False)
    self.position_embedder = nn.Embedding(max_len, n_hidden_dims[0])
    self.blocks = nn.ModuleList([])
    for i in range(n_hidden_layers):
      self.blocks.append(EncoderBlock(n_hidden_dims[i], int(n_hidden_dims[i]*intermediate_scale), n_heads, dropout_rate))
      if (i+1) % merge_after == 0 and (i+1) != n_hidden_layers:
        self.blocks.append(PatchMerge())

    self.out = nn.Linear(n_hidden_dims[-1], output_dim)

  def forward(self, x, n_rows, n_cols):
    patch_embeddings = self.patch_embedder(x)
    pos_embeddings = self.position_embedder(torch.arange(x.size(1), device=x.device))
    x = patch_embeddings + pos_embeddings
    for block in self.blocks:
      if block._get_name() == 'PatchMerge':
        x, n_rows, n_cols = block(x, n_rows, n_cols)
      else:
        x = block(x)

    x = F.gelu(self.out(x))
    return x, n_rows, n_cols


class PatchUnMerge(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, n_rows, n_cols):
    batch_size, n_patches, embed_dim = x.size()
    x = x.view(batch_size, n_rows, n_cols, embed_dim).permute(0, 3, 1, 2)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    x = x.permute(0, 2, 3, 1).flatten(1, 2)
    return x, n_rows*2, n_cols*2
    

class Generator(nn.Module):
  def __init__(self, input_dim, output_dim, max_len, n_hidden_layers, n_hidden_dims, merge_after, intermediate_scale, n_heads, dropout_rate):
    super().__init__()

    self.patch_embedder = nn.Linear(input_dim, n_hidden_dims[0], bias=False)
    self.position_embedder = nn.Embedding(max_len, n_hidden_dims[0])

    self.blocks = nn.ModuleList([])
    for i in range(n_hidden_layers):
      self.blocks.append(EncoderBlock(n_hidden_dims[i], int(n_hidden_dims[i]*intermediate_scale), n_heads, dropout_rate))
      if (i+1) % merge_after == 0 and (i+1) != n_hidden_layers:
        self.blocks.append(PatchUnMerge())
    self.fc = nn.Linear(n_hidden_dims[-1], output_dim)

  def forward(self, x, n_rows, n_cols):
    patch_embeddings = self.patch_embedder(x)
    pos_embeddings = self.position_embedder(torch.arange(x.size(1), device=x.device))
    x = patch_embeddings + pos_embeddings
    for block in self.blocks:
      if block._get_name() == 'PatchUnMerge':
        x, n_rows, n_cols = block(x, n_rows, n_cols)
      else:
        x = block(x)
    x = self.fc(x).sigmoid()
    return x, n_rows, n_cols


class Discriminator(nn.Module):
  def __init__(self, input_dim, max_len, n_hidden_layers, embed_dim, intermediate_scale, n_heads, dropout_rate):
    super().__init__()

    self.patch_embedder = nn.Linear(input_dim, embed_dim, bias=False)
    self.position_embedder = nn.Embedding(max_len, embed_dim)
    self.blocks = nn.ModuleList([])
    for i in range(n_hidden_layers):
      self.blocks.append(EncoderBlock(embed_dim, int(embed_dim*intermediate_scale), n_heads, dropout_rate))

    self.out = nn.Linear(embed_dim, 1)

  def forward(self, x):
    x = self.patch_embedder(x)
    x += self.position_embedder(torch.arange(x.size(1), device=x.device))
    for block in self.blocks:
      x = block(x)
    x = self.out(x).sigmoid()
    return x
    


class Codebook(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)  # QA (rohan): if we are detaching the q, down below, are we training these embeddings?
    self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

  def forward(self, x):
    batch_size, seq_len, embd_dim = x.shape
    x = x.flatten(0, 1).contiguous()


    a = torch.sum(x**2, dim=1, keepdim=True)
    b = torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0)
    ab2 = torch.matmul(x, self.embedding.weight.t())
    d = a + b - 2 * ab2
    indices = torch.argmin(d, dim=1)
    q = self.embedding(indices)

    # loss
    loss = torch.mean((x.detach() - q)**2) + 0.5*torch.mean((x - q.detach())**2)

    out = x + (q - x).detach()
    out = out.view(batch_size, seq_len, embd_dim)
    return out, loss


# compiling the enc-codebook-gen models into a single model
class VQGAN(nn.Module):
  def __init__(self, encoder, codebook, generator):
    super().__init__()
    self.encoder = encoder
    self.codebook = codebook
    self.generator = generator

  def forward(self, inp, n_rows, n_cols):
    enc_inp, n_rows, n_cols = self.encoder(inp, n_rows, n_cols)
    tok_inp, loss = self.codebook(enc_inp)
    fake_img, n_rows, n_cols = self.generator(tok_inp, n_rows, n_cols)


    # for x in range(fake_img.shape[1]):
    #   print('  ', x, ':', torch.all(torch.eq(fake_img[0, x, :], fake_img[:, x, :])))

    # print('fake_img:', fake_img.shape)


    return fake_img, loss