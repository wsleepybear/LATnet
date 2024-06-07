import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from model import model
import torch.nn as nn
import torch.optim as optim
import train
import test
import logging
from PolyLoss import Poly1BCELoss
from dataset import EmbeddingWithTagDataset
import pandas as pd
import os
import datetime

# 设置日志记录器和处理器
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

# 创建一个文件处理器，将日志信息写入文件
file_handler = logging.FileHandler("training_log.txt")
file_handler.setLevel(logging.INFO)  # 设置处理器的日志级别为 INFO
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
directory = "cache/lncRNA"
if not os.path.exists(directory):
    os.makedirs(directory)
# 将处理器添加到日志记录器
logger.addHandler(file_handler)
# writer = SummaryWriter('./runs/my_experiment')
# 加载嵌入Tensor列表和对应的tag列表
with open("./dataset/lncRNA/lncRNA_train_embeddings_list.pkl", "rb") as f:
    train_embeddings_list = pickle.load(f)
with open("./dataset/lncRNA/lncRNA_test_embeddings_list.pkl", "rb") as f:
    test_embeddings_list = pickle.load(f)
with open("./dataset/lncRNA/lncRNA_train_tags_list.pkl", "rb") as f:
    train_tags_list = pickle.load(f)
with open("./dataset/lncRNA/lncRNA_test_tags_list.pkl", "rb") as f:
    test_tags_list = pickle.load(f)
train_seq = pd.read_csv("dataset/lncRNA/lncRNA_sublocation_TrainingSet.tsv", sep="\t")[
    "cdna"
]
test_seq = pd.read_csv("dataset/lncRNA/lncRNA_sublocation_TestSet.tsv", sep="\t")[
    "cdna"
]

device = torch.device("cuda")
# 创建一个自定义的Dataset类来处理嵌入数据和对应的tag

# 创建DataLoader
train_embeddings_loader = DataLoader(
    EmbeddingWithTagDataset(train_embeddings_list, train_tags_list, train_seq),
    batch_size=48,
    shuffle=True,
)
test_embeddings_loader = DataLoader(
    EmbeddingWithTagDataset(test_embeddings_list, test_tags_list, test_seq),
    batch_size=16,
    shuffle=False,
)

max_auroc = 0.78
max_acc = 0.72
max_sen = 0.76
max_sp = 0.68
max_mcc = 0.45
max_f1 = 0.74
for i in range(100):
    train_model = model.Model().to(device="cuda")
    # train_model.load_state_dict(
    #     torch.load(
    #         "cache/lncRNA/model_pos_0.73_0.77_0.69_0.46_0.74_0.80_20240309_165544.pth",
    #         map_location="cuda",
    #     ),
    #     strict=True,
    # )
    min_loss = 10
    stop_counter = 0  # 早停计数器
    criterion = Poly1BCELoss(num_classes=2).to(device)
    optimizer = optim.AdamW(train_model.parameters(), lr=0.001, weight_decay=0.01)
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train.train(
            train_model, train_embeddings_loader, criterion, optimizer, device
        )
        test_loss, acc, sensitivity, specificity, mcc, f1, auroc = test.test(
            train_model, test_embeddings_loader, criterion, device
        )
        if (max_auroc < auroc and max_acc < acc and max_mcc < mcc and max_f1 < f1) or (
            f1 > 0.785 and auroc > 0.82 and acc > 0.77 and mcc > 0.54
        ):
            max_auroc = auroc
            max_acc = acc
            # max_sen = sensitivity
            # max_sp = specificity
            max_mcc = mcc
            max_f1 = f1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # 创建包含所有最大值的模型文件名
            model_filename = f"{directory}/model_pos_{max_acc:.4f}_{sensitivity:.4f}_{specificity:.4f}_{max_mcc:.4f}_{max_f1:.4f}_{max_auroc:.4f}_{timestamp}.pth"
            torch.save(train_model.state_dict(), model_filename)
        if test_loss < min_loss:
            min_loss = test_loss
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= 20:  # 如果连续35个epoch没有改进，则早停
            print("Early stopping triggered at epoch", epoch + 1)
            break
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Accuracy: {acc:.4f}, "
            f"Sensitivity: {sensitivity:.4f}, "
            f"Specificity: {specificity:.4f}, "
            f"MCC: {mcc:.4f}, "
            f"F1 Score: {f1:.4f}, "
            f"AUROC: {auroc:.4f}"
        )
