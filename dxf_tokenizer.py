import torch, torch.nn as nn, math

class Fourier(nn.Module):
    def __init__(self, in_dim=1, B=24, sigma=8.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, B) * sigma, requires_grad=False)
        
    def forward(self, x):
        y = 2*math.pi * x @ self.B
        return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)

class KindArgCrossAttnPool(nn.Module):
    def __init__(self, d_model, n_heads, n_kinds):
        super().__init__()
        self.kind_table = nn.Embedding(n_kinds, d_model)  # learned queries
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln  = nn.LayerNorm(d_model)

    def forward(self, slot_emb, slot_mask, kinds_present):
        """
        slot_emb:   [N, S, d]  # per-slot embeddings (argument embeddings)
        slot_mask:  [N, S]     # 1=used, 0=unused  -> becomes key_padding_mask
        kinds_present: [N, K]  # 1=kind appears in this primitive, 0=absent

        Returns: pooled [N, d], optional attn weights
        """
        N, S, d = slot_emb.shape
        N2, K = kinds_present.shape
        assert N == N2

        # Build queries from all kinds, then mask out absent ones when pooling
        kind_idx = torch.arange(K, device=slot_emb.device).unsqueeze(0).expand(N, K)
        q = self.kind_table(kind_idx)                       # [N, K, d]

        # MHA expects key_padding_mask: True = pad
        key_padding = (slot_mask == 0)                      # [N, S] bool

        out, attn = self.mha(q, slot_emb, slot_emb, key_padding_mask=key_padding)
        out = self.ln(out)                                  # [N, K, d]

        # Pool only over present kinds
        kp = kinds_present.unsqueeze(-1)                    # [N, K, 1]
        num = (out * kp).sum(dim=1)                         # [N, d]
        den = kp.sum(dim=1).clamp_min(1.0)                  # [N, 1]
        pooled = num / den                                  # [N, d]
        return pooled, attn                                 # attn: [N, K, S]

class FieldTypedProjector(nn.Module):
    def __init__(self, n_kinds, d_model, B=24):
        super().__init__()
        self.kind_emb = nn.Embedding(n_kinds, d_model)
        self.ff = Fourier(1, B=B)
        # one tiny MLP per kind
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2*B, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            for _ in range(n_kinds)
        ])

    def forward(self, values, kinds):
        N, S, _ = values.shape
        out = torch.zeros(N, S, self.kind_emb.embedding_dim, device=values.device)
        vals_ff = self.ff(values.view(-1,1))
        for k in range(self.kind_emb.num_embeddings):
            idx = (kinds.view(-1) == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0: continue
            out.view(-1, out.size(-1))[idx] = self.mlps[k](vals_ff[idx])
        return out + self.kind_emb(kinds)

class PrimitiveTokenizer(nn.Module):
    def __init__(self, n_kinds, n_types, n_layers=4096, d_model=256):
        super().__init__()
        self.field_proj = FieldTypedProjector(n_kinds, d_model)
        self.type_emb   = nn.Embedding(n_types, d_model)
        self.layer_emb  = nn.Embedding(n_layers, d_model)
        self.meta_lin   = nn.Linear(4, d_model)
        self.fuse       = nn.Sequential(nn.Linear(3*d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, values, kinds, mask, prim_type, layer_id, meta):
        N, S = values.shape
        h_slots = self.field_proj(values.unsqueeze(-1), kinds)
        h_slots = h_slots * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        h_num = h_slots.sum(dim=1) / denom

        h = torch.cat([
            h_num,
            self.type_emb(prim_type),
            self.layer_emb(layer_id)
        ], dim=-1)
        return self.fuse(h) + self.meta_lin(meta.float())
    
### cleaned code (WIP)
# converts one-hot vectors to a (learned) embedding
# for primitive type and argument type embedding
class Onehot2Embed(nn.Module):
    def __init__(self, n_classes, embed_dim):
        super.__init__()
        self.lookup_table = nn.Parameter(torch.randn(embed_dim, n_classes))
    
    def forward(self, types):
        return torch.transpose(torch.matmul(self.lookup_table, types), 1, 0)

class ArgumentEmbedding(nn.Module):
    def __init__(self, fourier_dim, embed_dim, num_classes):
        super.__init__()
        self.arg_type_embedding = Onehot2Embed(num_classes, embed_dim)
        self.fourier_features = Fourier(1, fourier_dim)
        # one tiny MLP per kind
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2*fourier_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
            for _ in range(num_classes)
        ])

    
