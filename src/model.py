import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRec(nn.Module):
    def __init__(self, num_items, hidden_dim=2048, max_len=50, num_heads=2, num_blocks=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Эмбеддинги
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        # Self-attention
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_seq):

        device = input_seq.device
        seq_len = input_seq.size(1)

        # item + position embeddings
        pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        item_emb = self.item_embedding(input_seq)
        pos_emb = self.position_embedding(pos)
        x = item_emb + pos_emb
        x = self.layer_norm(self.dropout(x))

        # Attention mask: запрещаем смотреть в будущее (causal mask)
        attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()

        # Transformer layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask=attn_mask)

        return x  # (B, T, D)

    def predict(self, seq_output, candidate_items):
        """
        seq_output: (B, D) — последнее скрытое состояние из последовательности
        candidate_items: (B, N) — список item_id для оценки
        Возвращает логиты для N кандидатов
        """
        item_emb = self.item_embedding(candidate_items)  # (B, N, D)
        seq_output = seq_output.unsqueeze(1)             # (B, 1, D)
        logits = torch.sum(seq_output * item_emb, dim=-1)  # (B, N)
        return logits