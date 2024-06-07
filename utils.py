import numpy as np
import torch

def create_positional_encoding(sequence_length, embedding_dim):
    position_enc = np.array(
        [
            [
                pos / np.power(10000, 2 * (j // 2) / embedding_dim)
                for j in range(embedding_dim)
            ]
            for pos in range(sequence_length)
        ]
    )

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # 2i+1
    return torch.tensor(position_enc, dtype=torch.float32, requires_grad=False)