import dataclasses
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class EncoderConfig:
  input_dim: int
  max_len: int
  n_hidden_layers: int
  n_hidden_dims: list
  merge_after: int
  intermediate_scale: int
  n_heads: int

@dataclasses.dataclass
class GeneratorConfig:
  input_dim: int
  max_len: int
  n_hidden_layers: int
  n_hidden_dims: list
  merge_after: int
  intermediate_scale: int
  n_heads: int

@dataclasses.dataclass
class DiscriminatorConfig:
  input_dim: int
  max_len: int
  n_hidden_layers: int
  embed_dim: int
  intermediate_scale: int
  n_heads: int

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
    q = F.gelu(self.q(x)).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    k = F.gelu(self.k(x)).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
    v = F.gelu(self.v(x)).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    attn = torch.einsum('bhid,bhjd->bhij', q, k) / self.head_dim**0.5
    attn = F.softmax(attn, dim=-1)
    x = torch.einsum('bhij,bhjd->bhid', attn, v).permute(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
    x = F.gelu(self.out(x))
    return x

class MLP(nn.Module):
  def __init__(self, embed_dim, intermediate_dim):
    super().__init__()
    self.fc1 = nn.Linear(embed_dim, intermediate_dim)
    self.fc2 = nn.Linear(intermediate_dim, embed_dim)

  def forward(self, x):
    return self.fc2(F.gelu(self.fc1(x)))

class EncoderBlock(nn.Module):
  def __init__(self, embed_dim, intermediate_dim, n_heads):
    super().__init__()
    self.attn_block = AttentionBlock(embed_dim, n_heads)
    self.mlp = MLP(embed_dim, intermediate_dim)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, x):
    x = x + self.attn_block(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x

class PatchMerge(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.fc = nn.Linear(4*embed_dim, 2*embed_dim)

  def forward(self, x, n_rows, n_cols):
    batch_size, n_patches, embed_dim = x.size()
    x = x.view(batch_size, n_rows, n_cols, embed_dim)
    tl = x[:, ::2, ::2, :]
    tr = x[:, ::2, 1::2, :]
    bl = x[:, 1::2, ::2, :]
    br = x[:, 1::2, 1::2, :]
    x = torch.cat([tl, tr, bl, br], dim=-1)
    x = x.view(batch_size, -1, 4*embed_dim)
    x = self.fc(x)
    return x, n_rows//2, n_cols//2



class Encoder(nn.Module):
  def __init__(self, input_dim, max_len, n_hidden_layers, n_hidden_dims, merge_after, intermediate_scale, n_heads):
    super().__init__()

    self.patch_embedder = nn.Linear(input_dim, n_hidden_dims[0], bias=False)
    self.position_embedder = nn.Embedding(max_len, n_hidden_dims[0])
    self.blocks = nn.ModuleList([])
    for i in range(n_hidden_layers):
      self.blocks.append(EncoderBlock(n_hidden_dims[i], int(n_hidden_dims[i]*intermediate_scale), n_heads))
      if (i+1) % merge_after == 0 and (i+1) != n_hidden_layers:
        self.blocks.append(PatchMerge(n_hidden_dims[i]))

  def forward(self, x, n_rows, n_cols):
    x = self.patch_embedder(x)
    x += self.position_embedder(torch.arange(x.size(1), device=x.device))
    for block in self.blocks:
      if block._get_name() == 'PatchMerge':
        x, n_rows, n_cols = block(x, n_rows, n_cols)
      else:
        x = block(x)
    return x, n_rows, n_cols


class PatchUnMerge(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    assert embed_dim % 4 == 0, "Embedding dimension must be divisible by 4"
    self.fc = nn.Linear(embed_dim // 4, embed_dim // 2)

  def forward(self, x, n_rows, n_cols):
    batch_size, n_patches, embed_dim = x.size()
    x = x.view(batch_size, n_rows, n_cols, 2, 2, embed_dim // 4)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, n_rows*2, n_cols*2, embed_dim // 4).flatten(1, 2)
    x = self.fc(x)
    return x, n_rows*2, n_cols*2
    

class Generator(nn.Module):
  def __init__(self, input_dim, max_len, n_hidden_layers, n_hidden_dims, merge_after, intermediate_scale, n_heads):
    super().__init__()

    self.blocks = nn.ModuleList([])
    for i in range(n_hidden_layers):
      self.blocks.append(EncoderBlock(n_hidden_dims[i], int(n_hidden_dims[i]*intermediate_scale), n_heads))
      if (i+1) % merge_after == 0 and (i+1) != n_hidden_layers:
        self.blocks.append(PatchUnMerge(n_hidden_dims[i]))
    self.fc = nn.Linear(n_hidden_dims[-1], input_dim)

  def forward(self, x, n_rows, n_cols):
    for block in self.blocks:
      if block._get_name() == 'PatchUnMerge':
        x, n_rows, n_cols = block(x, n_rows, n_cols)
      else:
        x = block(x)
    x = self.fc(x).sigmoid()
    return x, n_rows, n_cols


class Discriminator(nn.Module):
  def __init__(self, input_dim, max_len, n_hidden_layers, embed_dim, intermediate_scale, n_heads):
    super().__init__()

    self.patch_embedder = nn.Linear(input_dim, embed_dim, bias=False)
    self.position_embedder = nn.Embedding(max_len, embed_dim)
    self.blocks = nn.ModuleList([])
    for i in range(n_hidden_layers):
      self.blocks.append(EncoderBlock(embed_dim, int(embed_dim*intermediate_scale), n_heads))

    self.out = nn.Linear(embed_dim, 1)

  def forward(self, x):
    x = self.patch_embedder(x)
    x += self.position_embedder(torch.arange(x.size(1), device=x.device))
    for block in self.blocks:
      x = block(x)
    x = self.out(x)
    return x
    


class Codebook(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim

  def forward(self, x):
    batch_size, seq_len, embd_dim = x.shape
    # calculate euclidean distance between x and each embedding
    dist = torch.norm(x.unsqueeze(-2) - self.embedding.weight, dim=-1)
    indices = torch.argmin(dist, dim=-1).unsqueeze(-1)  # shape = (batch_size, seq_len, 1)
    # create ohe vector with indices == 1 and other == 0
    ohe = torch.zeros((batch_size, seq_len, self.num_embeddings), device=x.device)
    ohe.scatter_(-1, indices, 1)
    ohe = ohe.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
    out = (ohe * self.embedding.weight).sum(dim=-2)
    # to let the grads flow
    out = x + out - x.detach()
    return out


# compiling the enc-codebook-gen models into a single model
class VQGAN(nn.Module):
  def __init__(self, encoder, codebook, generator):
    super().__init__()
    self.encoder = encoder
    self.codebook = codebook
    self.generator = generator

  def forward(self, inp, n_rows, n_cols):
    enc_inp, n_rows, n_cols = self.encoder(inp, n_rows, n_cols)
    tok_inp = self.codebook(enc_inp)
    fake_img, n_rows, n_cols = self.generator(tok_inp, n_rows, n_cols)

    # loss calculation
    sg_enc_inp = enc_inp.detach()
    sg_tok_inp = tok_inp.detach()
    loss = torch.norm(sg_enc_inp - tok_inp) + torch.norm(sg_tok_inp - enc_inp)
    return fake_img, loss