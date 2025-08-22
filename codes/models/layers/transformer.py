import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )

        self.projection_dim = embed_dim // num_heads

        # Linear layers (équivalents de Dense de Keras)
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense   = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        """
        query, key, value: (batch, num_heads, seq_len, projection_dim)
        """
        # produit scalaire QK^T
        score = torch.matmul(query, key.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        dim_key = key.size(-1)
        scaled_score = score / math.sqrt(dim_key)  # scaling
        weights = F.softmax(scaled_score, dim=-1)  # softmax sur la dernière dimension
        output = torch.matmul(weights, value)      # (batch, num_heads, seq_len, projection_dim)
        return output, weights

    def separate_heads(self, x, batch_size):
        """
        Transformer un tenseur (batch, seq_len, embed_dim)
        en (batch, num_heads, seq_len, projection_dim)
        """
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inputs):
        """
        inputs: (batch_size, seq_len, embed_dim)
        """
        batch_size = inputs.size(0)

        # projections linéaires
        query = self.query_dense(inputs)
        key   = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # reshape + permutation pour séparer les têtes
        query = self.separate_heads(query, batch_size)
        key   = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # appliquer l'attention multi-têtes
        attention, weights = self.attention(query, key, value)

        # remettre les têtes ensemble : (batch, seq_len, embed_dim)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = attention.view(batch_size, -1, self.embed_dim)

        # projection finale
        output = self.combine_heads(concat_attention)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ff_dim=2048, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.dense1 = nn.Linear(embed_dim, ff_dim)
        self.dense2 = nn.Linear(ff_dim, embed_dim)
        
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Dropout
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        """
        inputs: (batch_size, seq_len, embed_dim)
        Utilise automatiquement self.training pour les dropouts
        """
        # Multi-head self-attention
        attn_output = self.att(inputs)                       # (batch, seq_len, embed_dim)
        attn_output = self.dropout1(attn_output)             # Dropout automatique selon self.training
        out1 = self.layernorm1(inputs + attn_output)        # Add & Norm

        # Feed-forward
        ffn_output = F.relu(self.dense1(out1))
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output)               # Dropout automatique selon self.training

        # Add & Norm
        return self.layernorm2(out1 + ffn_output)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices; cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]  # shape (1, position, d_model)
    return torch.tensor(pos_encoding, dtype=torch.float32)  # float32 tensor

class TransformerBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, maximum_position_encoding, num_heads=8, ff_dim=2048, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Position encoding buffer (non-trainable)
        self.register_buffer('pos_encoding', positional_encoding(maximum_position_encoding, embed_dim))
        
        # Encoder layers
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(rate)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        # scale embeddings and add positional encoding
        x = x * math.sqrt(self.embed_dim)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # pass through encoder layers
        for layer in self.enc_layers:
            x = layer(x)  # Plus besoin de passer training explicitement
        return x  # (batch_size, seq_len, embed_dim)