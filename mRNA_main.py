import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from model import mRNA_model
import torch.nn as nn
import torch.optim as optim
import train
import test
import logging
from PolyLoss import Poly1BCELoss
from mRNA_dataset import EmbeddingWithTagDataset
import pandas as pd
import os


logger = logging.getLogger('training_logger')
logger.setLevel(logging.INFO)  

file_handler = logging.FileHandler('training_log.txt')
file_handler.setLevel(logging.INFO) 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
directory = "cache/mRNA"
if not os.path.exists(directory):
    os.makedirs(directory)
logger.addHandler(file_handler)

with open("./dataset/mRNA/mRNA_train_embeddings_list.pkl", "rb") as f:
    train_embeddings_list = pickle.load(f)
with open("./dataset/mRNA/mRNA_test_embeddings_list.pkl", "rb") as f:
    test_embeddings_list = pickle.load(f)
with open("./dataset/mRNA/mRNA_train_tags_list.pkl", "rb") as f:
    train_tags_list = pickle.load(f)
with open("./dataset/mRNA/mRNA_test_tags_list.pkl", "rb") as f:
    test_tags_list = pickle.load(f)
train_seq = pd.read_csv("dataset/mRNA/mRNA_sublocation_TrainingSet.tsv", sep="\t")[
    "cdna"
]
test_seq = pd.read_csv("dataset/mRNA/mRNA_sublocation_TestSet.tsv", sep="\t")["cdna"]

device = torch.device("cuda")

train_embeddings_loader = DataLoader(
    EmbeddingWithTagDataset(train_embeddings_list, train_tags_list, train_seq),
    batch_size=24,
    shuffle=True,
)
test_embeddings_loader = DataLoader(
    EmbeddingWithTagDataset(test_embeddings_list, test_tags_list, test_seq),
    batch_size=8,
    shuffle=True,
)


train_model = mRNA_model.Model().to(device='cuda')

criterion = Poly1BCELoss(num_classes=2).to(device)
optimizer = optim.AdamW(train_model.parameters(), lr=0.001, weight_decay=0.01)
num_epochs = 150
min_loss = 10
stop_counter = 0 

for epoch in range(num_epochs):
    train_loss=train.train(train_model,train_embeddings_loader,criterion,optimizer,device)
    test_loss, acc, sensitivity, specificity, mcc, f1, auroc=test.test(train_model,test_embeddings_loader,criterion,device)

    if test_loss < min_loss:
        min_loss = test_loss
        stop_counter = 0
    else:
        stop_counter += 1

    if stop_counter >= 35: # 如果连续35个epoch没有改进，则早停
        print("Early stopping triggered at epoch", epoch + 1)
        break

        
