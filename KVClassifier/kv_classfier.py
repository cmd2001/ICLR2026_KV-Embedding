import os
import torch
from torch import nn

# target: build a network, input: [Batch, layer_idx, Head, Seq, Dim], output: [Batch]

class KVClassifier(nn.Module):
    """
    kv_part:   "k" | "v" | "kv_cat"
    pos_agg:   "mean" | "last" | "cls"      (aggregate across sequence)
    head_agg:  "mean" | "flatten"           (aggregate across heads)
    layer_agg: "mean" | "sum" | "flatten"   (aggregate across layers)
    """
    def __init__(self, n_layers, n_heads, seq_len, n_dim, hidden_dim=512, dtype=torch.bfloat16,
                 kv_part="kv_cat",
                 pos_agg="sum",
                 head_agg="flatten",
                 layer_agg="mean",
                 normalize_embeddings=False):
        super(KVClassifier, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.n_dim = n_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        
        self.kv_part = kv_part
        self.pos_agg = pos_agg
        self.head_agg = head_agg
        self.layer_agg = layer_agg
        self.normalize_embeddings = normalize_embeddings
        
        self.input_dim = 1
        if kv_part == "kv_cat":
            self.input_dim *= 2
        if head_agg == "flatten":
            self.input_dim *= n_heads
        if pos_agg == "flatten":
            self.input_dim *= seq_len
        if layer_agg == "flatten":
            self.input_dim *= n_layers
        self.input_dim *= n_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
        )
    def forward(self, k_cache, v_cache):
        if self.pos_agg == "mean": # mean over sequence
            k_cache = k_cache.mean(dim=3)
            v_cache = v_cache.mean(dim=3)
        elif self.pos_agg == "sum":
            k_cache = k_cache.sum(dim=3)
            v_cache = v_cache.sum(dim=3)
        elif self.pos_agg == "last": # last token
            k_cache = k_cache[:, :, :, -1, :]
            v_cache = v_cache[:, :, :, -1, :]
        elif self.pos_agg == "cls": # first token
            k_cache = k_cache[:, :, :, 0, :]
            v_cache = v_cache[:, :, :, 0, :]
        else:
            raise ValueError(f"Unknown pos_agg: {self.pos_agg}")
        
        if self.head_agg == "mean": # mean over heads
            k_cache = k_cache.mean(dim=2)
            v_cache = v_cache.mean(dim=2)
        
        if self.layer_agg == "mean": # mean over layers
            k_cache = k_cache.mean(dim=1)
            v_cache = v_cache.mean(dim=1)
        elif self.layer_agg == "sum": # sum over layers
            k_cache = k_cache.sum(dim=1)
            v_cache = v_cache.sum(dim=1)
        
        k_cache = k_cache.reshape(k_cache.shape[0], -1)
        v_cache = v_cache.reshape(v_cache.shape[0], -1)
        
        # print(k_cache.shape, v_cache.shape)
        if self.kv_part == "k":
            review = k_cache
        elif self.kv_part == "v":
            review = v_cache
        elif self.kv_part == "kv_cat": # cat along feature dim, not batch dim
            review = torch.cat([k_cache, v_cache], dim=-1)
        else:
            raise ValueError(f"Unknown kv_part: {self.kv_part}")
        # print("review shape:", review.shape)
        
        if self.normalize_embeddings:
            review = review / (review.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        output = self.fc(review)
        return output.squeeze(-1)

    def backward(self, loss):
        loss.backward()