import torch
from torch.utils.data import Dataset

embedding_dict = {
    "A": [1, 0, 0, 0, 0.1260, 1, 1, 1],
    "T": [0, 1, 0, 0, 0.1335, 0, 0, 1],
    "C": [0, 0, 1, 0, 0.1340, 0, 1, 0],
    "G": [0, 0, 0, 1, 0.0806, 1, 0, 0],
}
start_embed = [0, 0, 0, 0, 0, 0, 0, 0]
end_embed = [0, 0, 0, 0, 0, 0, 0, 0]


class EmbeddingWithTagDataset(Dataset):
    def __init__(self, embeddings_list, tags_list, seq_list):
        self.embeddings_list = embeddings_list
        self.tags_list = tags_list
        embedded_sequences = seq_list.apply(
            lambda seq: [embedding_dict[base] for base in seq]
        )
        self.seq_list = embedded_sequences.apply(process_sequence)
        # self.kmer_freq = kmer_freq
        padded_embeddings_list = pad_embeddings(self.embeddings_list)

        # 现在 self.embeddings_list 已经被填充，并且每个 embedding 的大小都是 3000
        self.embeddings_list = padded_embeddings_list

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        # 返回嵌入Tensor和对应的tag
        return self.embeddings_list[idx], self.tags_list[idx], self.seq_list[idx]


def process_sequence(seq):
    # 截断或补齐序列
    if len(seq) > 2998:
        # 如果超过2998，则截取前1499和后1499
        seq = seq[:1499] + seq[-1499:]
    else:
        # 如果不足2998，则在结尾处进行零填充
        padding = [[0, 0, 0, 0, 0, 0, 0, 0]] * (2998 - len(seq))
        seq = seq + padding

    # 添加开头和结尾的嵌入标志
    seq = [start_embed] + seq + [end_embed]

    # 确保最终长度为3000
    assert len(seq) == 3000
    return torch.tensor(seq)


# 定义一个函数来添加零填充
def pad_embeddings(embeddings_list, target_size=3000):
    padded_embeddings_list = []
    for embedding in embeddings_list:
        # 计算需要添加的零填充数量
        padding_size = target_size - embedding.size(0)
        if padding_size > 0:
            # 创建零填充张量
            padding = torch.zeros(padding_size, embedding.size(1))
            # 将零填充张量和原始 embedding 拼接起来
            padded_embedding = torch.cat((embedding, padding), dim=0)
        else:
            # 如果 embedding 大小已经达到或超过目标大小，不需要填充
            padded_embedding = embedding
        padded_embeddings_list.append(padded_embedding)
    return padded_embeddings_list
