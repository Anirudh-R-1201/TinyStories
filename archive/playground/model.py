from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self, d, H, T, bias=False, dropout=0.2):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0

        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3*d, bias=bias)

        # projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)

        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d

        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(T, T))
                                    .view(1, 1, T, T))

    def forward(self, x):
        B, T, _ = x.size() # batch size, sequence length, embedding dimensionality

        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v  = self.c_attn(x).split(self.d, dim=2) # [B, T, d]

        # reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2) # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        # compute the attention matrix, perform masking, and apply dropout
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # [B, H, T, T]
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # compute output vectors for each token
        y = att @ v # [B, H, T, d // H]

        # concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # first linear layer
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        # activation function
        self.gelu    = nn.GELU(approximate='tanh')
        # second linear layer
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # apply first linear layer
        x = self.c_fc(x)
        # apply activation function
        x = self.gelu(x)
        # apply second linear layer
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # layer normalization before attention
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # self-attention layer
        self.attn = CausalSelfAttention(config)
        # layer normalization before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # MLP layer
        self.mlp = MLP(config)

    def forward(self, x):
        # apply layer normalization and self-attention, then add residual connection
        x = x + self.attn(self.ln_1(x))
        # apply layer normalization and MLP, then add residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # final layer normalization
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initialize parameters
        self.apply(self._init_weights)
    
    def forward(self, idx):
        #idx and targets are both (B, T) tensor of integers
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_embeddings = self.transformer.wpe(pos) # shape (T, n_embd)
        token_embeddings = self.transformer.wte(idx) # shape (B, T, n_embd)
        x = token_embeddings + pos_embeddings
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
