import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
)
import torch.nn.functional as F
import numpy as np


def test(model, test_loader, criterion, device):
    model.eval()
    all_targets = []
    all_predictions = []
    all_outputs = []
    total_loss = 0.0
    with torch.no_grad():
        for data, target,seq in test_loader:
            data, target,seq = data.to(device), target.float().to(device),seq.float().to(device)

            target_onehot = F.one_hot((target >= 0.5).to(torch.int64), 2).to(
                device=device
            )
            output = model(data,seq)
            loss = criterion(output, target)
            total_loss += loss.item()
            output = nn.Softmax(dim=1)(output)
            predictions = (output >= 0.5).float()
            all_targets.extend(target_onehot.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())

    acc = accuracy_score(all_targets, all_predictions)
    true_classes = np.argmax(all_targets, axis=1)
    pred_classes = np.argmax(all_predictions, axis=1)
    tn, fp, fn, tp = confusion_matrix(true_classes, pred_classes).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(true_classes, pred_classes)
    f1 = f1_score(true_classes, pred_classes)
    auroc = roc_auc_score(all_targets, all_outputs)

    avg_loss = total_loss / len(test_loader)
    return avg_loss, acc, sensitivity, specificity, mcc, f1, auroc
