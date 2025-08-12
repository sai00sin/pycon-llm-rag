import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=384, n_head=6, attn_p=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_p)
        self.resid_drop = nn.Dropout(attn_p)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # (B,T,h,d)
        q = q.transpose(1,2)  # (B,h,T,d)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1,1,T,T)
        att = att.masked_fill(mask==0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B,h,T,d)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model=384, n_head=6, mlp_ratio=4, p=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, attn_p=p)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio*d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio*d_model, d_model),
            nn.Dropout(p)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=512, n_layer=6, d_model=384, n_head=6, p=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, p=p) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_len
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
