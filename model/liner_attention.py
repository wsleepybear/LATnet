import torch.nn as nn
import torch
class LinearAttention(nn.Module):
    def __init__(self, n=None, slopes=None, device=None):
        super(LinearAttention, self).__init__()
        self.n = n
        self.slopes = slopes if slopes is not None else [1.0]
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_mask(self, n, slope):
        mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
        for i in range(n):
            x = torch.arange(i + 1)
            y = slope * x
            mask[i, :i + 1] = -torch.flip(y, [0])
        return torch.exp(mask)

    def get_full_mask(self, n, slopes):
        arr = []
        for slope in slopes:
            arr.append(self.get_mask(n, slope))
        mask = torch.stack(arr, dim=0)
        return mask

    def linear_attn(self, q, k, v):
        b, h, n, d = q.shape
        mask = self.get_full_mask(n, self.slopes).to(q.device).to(torch.float32)
        qk = torch.matmul(q, k.transpose(2, 3))
        qk = (qk.to(torch.float32) * mask).to(q.dtype)
        o = torch.matmul(qk, v)
        return o

    def forward(self, q, k, v):
        return self.linear_attn(q, k, v)
