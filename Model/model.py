import math
import random
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.utils.checkpoint as checkpoint
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torchmetrics
from torch import nn
from torchmetrics import ConfusionMatrix, Accuracy, F1Score, Recall
from torch.utils.data import Dataset
import sklearn


class FNN_Model(nn.Module):
    def __init__(self, dropout):
        super(FNN_Model, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(102, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.output(x)

        return x


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        self.feature0 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.adaptpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.adaptpool2 = nn.AdaptiveAvgPool1d(2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feature0(x)
        x = self.adaptpool1(x)
        first_x = torch.flatten(x, start_dim=1)
        x = self.adaptpool2(first_x)

        outputDir = {"embedding": first_x, "output": x}

        return outputDir


# Supervised learning on CSD
def SL_train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    optimizer.zero_grad()

    # LOSS function
    task_ce_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.4 ,0.6])).to(device)

    # Metrix
    acclosses = AverageMeter()
    train_accuracy = Accuracy().to(device)
    confmatCOM = ConfusionMatrix(num_classes=2, normalize='true').to(device)

    for batch, (data, labels) in enumerate(data_loader):
        data, labels = data.float().to(device), labels.long().to(device)

        optimizer.zero_grad()
        outputs = model(data)["output"]
        loss = task_ce_criterion(outputs, labels)

        acclosses.update(loss.item(), data.shape[0])

        loss.backward()
        optimizer.step()

        # Metrix
        _, outputs = torch.max(outputs.data, dim=1)
        train_accuracy(outputs, labels)
        confmatCOM(outputs, labels)

    # Metrix Sum
    train_loss = acclosses.avg
    train_acc = train_accuracy.compute().cpu().item()
    confmat = confmatCOM.compute().cpu().numpy()
    TP, FN = confmat[0, 0], confmat[0, 1]
    FP, TN = confmat[1, 0], confmat[1, 1]

    # 计算Recall, Precision, F1, FPR
    train_Recall = TP / (TP + FN)
    train_Precision = TP / (TP + FP)
    train_F1 = 2 * train_Precision * train_Recall / (train_Precision + train_Recall)
    train_FPR = FN / (TP + FN)

    train_accuracy.reset()
    confmatCOM.reset()

    return train_loss, train_acc, train_Recall, train_Precision, train_F1, train_FPR, confmat




@torch.no_grad()
def updata_EMA_model(EMA_model, model, keep_rate=0.996):
    new_EMA_model_dict = EMA_model.state_dict()
    model_dict = model.state_dict()

    for key, value in EMA_model.state_dict().items():
        if key in model_dict.keys():
            new_EMA_model_dict[key] = (
               model_dict[key] * (1 - keep_rate) + value * keep_rate
            )

    EMA_model.load_state_dict(new_EMA_model_dict)




# Scene adaptation
def SCENE_Apdation_train_one_epoch(EMA_model, model, optimizer, train_labeled_dataloader, train_unlabeled_dataloader,
                        confidence_threshold, device):
    EMA_model.train()
    model.train()
    optimizer.zero_grad()

    # Loss Function
    task_ce_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.4 ,0.6])).to(device)

    # Metrix
    labeled_acclosses = AverageMeter()
    unlabled_acclosses = AverageMeter()
    all_acclosses = AverageMeter()
    train_accuracy = Accuracy().to(device)
    confmatCOM = ConfusionMatrix(num_classes=2).to(device)

    train_unlabeled_iterator = iter(train_unlabeled_dataloader)

    for batch, (data, labels) in enumerate(train_labeled_dataloader):
        # Acq both labeled and unlabeled data
        data, labels = data.float().to(device), labels.long().to(device)

        try:
            unlabeled_data, unlabeled_labels = next(train_unlabeled_iterator)
        except StopIteration:
            train_unlabeled_iterator = iter(train_unlabeled_dataloader)
            unlabeled_data, unlabeled_labels = next(train_unlabeled_iterator)
        unlabeled_data, unlabeled_labels = unlabeled_data.float().to(device), unlabeled_labels.long().to(device)

        # generate the pseudo-label using teacher model, and use confidence-based Filter
        optimizer.zero_grad()
        with torch.no_grad():
            output = EMA_model(unlabeled_data)["output"]
            probs  = torch.nn.functional.softmax(output, dim=1)
            conf, classes = torch.max(probs, 1)

        IndexList = []
        for i in range(conf.shape[0]):
            if conf[i] > confidence_threshold:
                IndexList.append(i)


        try:
            unlabeled_data = unlabeled_data[IndexList]
            unlabeld_pseudo_label = classes[IndexList]

            # Learning based on CSD
            outputs = model(data)["output"]
            labeled_loss = task_ce_criterion(outputs, labels)
            labeled_acclosses.update(labeled_loss.item(), data.shape[0])

            # Learning based on UCD with pseudo_label
            unlabeled_outputs = model(unlabeled_data)["output"]
            unlabeled_loss = task_ce_criterion(unlabeled_outputs, unlabeld_pseudo_label)
            unlabled_acclosses.update(unlabeled_loss.item(), unlabeled_data.shape[0])

            loss = labeled_loss + unlabeled_loss
            all_acclosses.update(labeled_loss.item(), (data.shape[0] + unlabeled_data.shape[0]))

            loss.backward()
            optimizer.step()

            # EMA-model updata
            updata_EMA_model(EMA_model, model)

            # Metrix - ALL train data
            _, outputs = torch.max(outputs.data, dim=1)
            train_accuracy(outputs, labels)
            confmatCOM(outputs, labels)

            _, unlabeled_outputs = torch.max(unlabeled_outputs.data, dim=1)
            train_accuracy(unlabeled_outputs, unlabeled_labels[IndexList])
            confmatCOM(unlabeled_outputs, unlabeled_labels[IndexList])
        except StopIteration:
            print(f"without training")

    sl_loss = labeled_acclosses.avg
    ssl_loss = unlabled_acclosses.avg
    all_loss = all_acclosses.avg

    train_acc = train_accuracy.compute().cpu().item()
    confmat = confmatCOM.compute().cpu().numpy()
    TP, FN = confmat[0, 0], confmat[0, 1]
    FP, TN = confmat[1, 0], confmat[1, 1]

    train_Recall = TP / (TP + FN)
    train_Precision = TP / (TP + FP)
    train_F1 = 2 * train_Precision * train_Recall / (train_Precision + train_Recall)
    train_FPR = FN / (TP + FN)

    train_accuracy.reset()
    confmatCOM.reset()

    return sl_loss, ssl_loss, all_loss, train_acc, train_Recall, train_Precision, train_F1, train_FPR, confmat




# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # Loss
    task_ce_criterion = nn.CrossEntropyLoss().to(device)
    # Metrix
    acclosses = AverageMeter()  # 损失累积
    test_accuracy = Accuracy().to(device)
    confmatCOM = ConfusionMatrix(num_classes=2, normalize='true').to(device)

    for batch, (data, labels) in enumerate(data_loader):
        data, labels = data.float().to(device), labels.long().to(device)

        outputs = model(data)["output"]
        loss = task_ce_criterion(outputs, labels)

        acclosses.update(loss.item(), data.shape[0])
        _, outputs = torch.max(outputs.data, dim=1)
        test_accuracy.update(outputs, labels)
        confmatCOM.update(outputs, labels)

    test_loss = acclosses.avg
    test_acc = test_accuracy.compute().cpu().item()
    confmat = confmatCOM.compute().cpu().numpy()
    TP, FN = confmat[0, 0], confmat[0, 1]
    FP, TN = confmat[1, 0], confmat[1, 1]

    # 计算Recall, Precision, F1, FPR
    test_Recall = TP / (TP + FN)
    test_Precision = TP / (TP + FP)
    test_F1 = 2 * test_Precision * test_Recall / (test_Precision + test_Recall)
    test_FPR = FN / (TP + FN)

    test_accuracy.reset()
    confmatCOM.reset()

    return test_loss, test_acc, test_Recall, test_Precision, test_F1, test_FPR, confmat



def modelMetrics(trueLabel, predictLabel):

    sklearn_Recall = sklearn.metrics.recall_score(trueLabel, predictLabel, pos_label=0)
    sklearn_Acc = sklearn.metrics.accuracy_score(trueLabel, predictLabel)
    sklearn_Matrix = sklearn.metrics.confusion_matrix(trueLabel, predictLabel)
    sklearn_f1 = sklearn.metrics.f1_score(trueLabel, predictLabel, pos_label=0)
    sklearn_pre = sklearn.metrics.precision_score(trueLabel, predictLabel, pos_label=0)

    TP, FN = sklearn_Matrix[0, 0], sklearn_Matrix[0, 1]
    FP, TN = sklearn_Matrix[1, 0], sklearn_Matrix[1, 1]

    FPR = FN / (TP + FN)

    return sklearn_Acc, sklearn_Recall, sklearn_pre, sklearn_f1, FPR, sklearn_Matrix



class DatasetLoad(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)




class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


