import torch
from torch.utils.data import Dataset

embedding_dict = {
    "A": [1, 0, 0, 0, 0.1260, 1, 1, 1],
    "T": [0, 1, 0, 0, 0.1335, 0, 0, 1],
    "C": [0, 0, 1, 0, 0.1340, 0, 1, 0],
    "G": [0, 0, 0, 1, 0.0806, 1, 0, 0],
}


class EmbeddingWithTagDataset(Dataset):
    def __init__(self, embeddings_list, tags_list, seq_list):
        self.embeddings_list = embeddings_list
        self.tags_list = tags_list
        self.embedded_sequences = seq_list.apply(
            lambda seq: [embedding_dict[base] for base in seq]
        )

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        # 返回嵌入Tensor和对应的tag
        return (
            process_sequence(self.embeddings_list[idx], flag=True),
            self.tags_list[idx],
            process_sequence(torch.tensor(self.embedded_sequences[idx])),
        )


def process_sequence(seq, target_size=6000, flag=False):
    # 当flag为True时，不需要添加开头和结尾的嵌入标志，并且只在尾部进行补齐
    if flag:
        if seq.size(0) < target_size:
            padding_size = target_size - seq.size(0)
            padding = torch.zeros(padding_size, seq.size(1))
            seq = torch.cat((seq, padding), dim=0)  # 只在尾部补齐
        assert seq.size(0) == target_size
    else:
        # flag为False时，处理添加开头和结尾的嵌入标志
        if seq.size(0) >= target_size - 2:
            seq = torch.cat(
                (seq[: target_size // 2 - 1], seq[-(target_size // 2 - 1) :]), dim=0
            )
        else:
            padding_size = target_size - seq.size(0) - 2
            padding = torch.zeros(padding_size, seq.size(1))
            seq = torch.cat((seq, padding), dim=0)

        # 添加开头和结尾的嵌入标志
        start_embed = torch.zeros(1, seq.size(1))
        end_embed = torch.zeros(1, seq.size(1))
        seq = torch.cat((start_embed, seq, end_embed), dim=0)

    assert seq.size(0) == target_size
    return seq
