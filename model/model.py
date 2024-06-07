import torch
import torch.nn as nn
from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor
import numpy as np
from utils import create_positional_encoding

class LightningAttention(nn.Module):
    def __init__(self):
        super(LightningAttention, self).__init__()

    def forward(self, q, k, v, s):
        o = lightning_attn_func(q, k, v, s)

        return o


class LightningTransformer(nn.Module):
    def __init__(self, num_heads, sequence_length, embedding_dim):
        super(LightningTransformer, self).__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        head_dim = self.embedding_dim // self.num_heads
        self.sequence_length = sequence_length
        self.layers = nn.ModuleList([])
        self.ff = FeedForward(dim=head_dim, mult=6, dropout=0.4)

        self.attention = lightning_attn_func
        self.to_logits = nn.Sequential(
            nn.LayerNorm(head_dim),
        )
        self.norm = nn.LayerNorm(head_dim)
        self.drop = nn.Dropout(0.4)
        self.positional_encoding = create_positional_encoding(3000, 640)

    def forward(self, x):

        x = x.unsqueeze(1)
        head_dim = self.embedding_dim // self.num_heads
        x = self.norm(x.view(x.size(0), self.num_heads, -1, head_dim))
        residual = x
        for _ in range(2):
            x = self.norm(
                self.attention(
                    x,
                    x,
                    x,
                    _build_slope_tensor(self.num_heads).to(x.device).to(torch.float32),
                )
                + residual
            )
            x = self.norm(self.ff(x) + residual)
        output = torch.mean(x, dim=1)
        return output



class LightningTransformerseq(nn.Module):
    def __init__(self, num_heads, sequence_length, embedding_dim):
        super(LightningTransformerseq, self).__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        head_dim = self.embedding_dim // self.num_heads
        self.sequence_length = sequence_length
        self.layers = nn.ModuleList([])
        self.ff = FeedForward(dim=head_dim, mult=4, dropout=0.4)

        self.attention = lightning_attn_func
        self.to_logits = nn.Sequential(
            nn.LayerNorm(head_dim),
        )
        self.norm = nn.LayerNorm(head_dim)
        self.drop = nn.Dropout(0.4)
        self.positional_encoding = create_positional_encoding(
            sequence_length, embedding_dim
        )

    def forward(self, x):

        x = x.unsqueeze(1)
        x += self.positional_encoding[: x.size(1), :].unsqueeze(0).to(x.device)

        head_dim = self.embedding_dim // self.num_heads
        x = self.norm(x.view(x.size(0), self.num_heads, -1, head_dim))

        residual = x
        for _ in range(2):
            x = self.norm(
                self.attention(
                    x,
                    x,
                    x,
                    _build_slope_tensor(self.num_heads).to(x.device).to(torch.float32),
                )
                + residual
            )
            x = self.norm(self.ff(x) + x)
        output = torch.mean(x, dim=1)
        return output



class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *kwargs):
        x = self.norm(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class TextCNN(nn.Module):
    def __init__(self, input_dim=64, contextSizeList=[3, 4, 5], dropout=0.1):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_dim,  
                        out_channels=input_dim,  
                        kernel_size=ks, 
                    ),
                    nn.LeakyReLU(),
                    SELayer(input_dim),
                    nn.AdaptiveMaxPool1d(1),
                )
                for ks in contextSizeList
            ]
        )

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_dim * 3, input_dim),  
            nn.LeakyReLU(),  
            nn.Linear(input_dim, 8),  
        )  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        conv_outputs = [conv(x) for conv in self.convs]

        x = torch.cat(conv_outputs, 1)
        x = x.view(-1, x.size(1))

        x = self.dropout(x)
        out = self.fc(x)
        return out


class LastLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(16, 8),  
            nn.LeakyReLU(),  
            nn.Linear(8, 2),  
        )  

    def forward(self, x):
        x = self.fc(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.LightningTransformer = LightningTransformer(10, 3000, 640)
        self.LightningTransformerseq = LightningTransformerseq(4, 3000, 64)
        self.fc = nn.Sequential(
            nn.Linear(8, 32),  
            nn.LeakyReLU(),  
            nn.Linear(32, 64), 
        )  
        self.TextCNN = TextCNN()
        self.TextCNNseq = TextCNN(16)
        self.sigmoid = nn.Sigmoid()
        self.LastLayer = LastLayer()

    def forward(self, x, seqx):
        seqx = self.fc(seqx)
        x = self.LightningTransformer(x)

        seqx = self.LightningTransformerseq(seqx)
        x = self.TextCNN(x)
        seqx = self.TextCNNseq(seqx)

        x = torch.cat((x, seqx), dim=1)
        x = self.LastLayer(x)
        return x
