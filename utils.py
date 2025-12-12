import random
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
import csv
import torch
import torch.nn.functional as F
from torch import nn

class CosineLoss(nn.Module):
    def __init__(self, flags, xent=.1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        self.y = torch.Tensor([1]).to(flags.device)
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y,
                                              reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)
        return cosine_loss + self.xent * cent_loss

def cosine_loss(inputs, targets, reduce=True, device='cuda'):
    """
    :param inputs:
    :param targets: Must be LongTensor
    :param reduce: if True, average batches' losses
    :return:
    """
    norm_inputs = F.normalize(inputs, p=2, dim=1)
    one_hot_y = F.one_hot(targets, num_classes=2).type(torch.FloatTensor).to(device)
    cosine_sim = torch.einsum('ij,ij->i', norm_inputs, one_hot_y)
    loss = 1-cosine_sim
    if reduce==True:
        loss = loss.mean()
    return loss

def sample(iterator, k):
    """
    Samples k elements from an iterable object.

    :param iterator: an object that is iterable
    :param k: the number of items to sample
    """
    # fill the reservoir to start
    result = [next(iterator) for _ in range(k)]

    n = k - 1
    for item in iterator:
        n += 1
        s = random.randint(0, n)
        if s < k:
            result[s] = item

    return result

def compute_accuracy(labels, predictions):
    # correct = torch.eq(predictions, labels).sum().item()
    # accuracy = accuracy_score(y_true=np.argmax(labels, axis=-1), y_pred=np.argmax(predictions, axis=-1))
    accuracy = accuracy_score(labels, predictions)
    return accuracy

def metric(label, predict, prob):
    tn, fp, fn, tp = confusion_matrix(label, predict).ravel()
    total = tn + fp + fn + tp
    acc = (tn + tp) / total
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc = roc_auc_score(label, prob)
    return auc, acc, sen, spec

def decision(sw_label, sw_pred, class_n, sw_per_sub):
    sw_label = np.array(sw_label).flatten()
    sw_pred = np.array(sw_pred).flatten()
    label = np.zeros(shape=(int(sw_label.shape[0] / sw_per_sub)), dtype=np.int8)
    pred = np.zeros(shape=(int(sw_pred.shape[0] / sw_per_sub)), dtype=np.int8)
    for i in range(0, int(sw_label.shape[0] / sw_per_sub)):
        counts1 = np.bincount(sw_label[sw_per_sub * i:sw_per_sub * (i + 1)], minlength=class_n)
        counts2 = np.bincount(sw_pred[sw_per_sub * i:sw_per_sub * (i + 1)], minlength=class_n)
        ind1 = np.argmax(counts1)
        ind2 = np.argmax(counts2)
        label[i] = ind1
        pred[i] = ind2
        # print(label, pred)
    return label, pred

def soft_decision(sw_label, sw_pred, class_n, sw_per_sub):
    sw_label = np.array(sw_label).flatten()
    # sw_pred0 = np.array(sw_pred[:, 0]).flatten()
    # sw_pred1 = np.array(sw_pred[:, 1]).flatten()
    label = np.zeros(shape=(int(sw_label.shape[0] / sw_per_sub)), dtype=np.float)
    prob = np.zeros(shape=(int(sw_pred.shape[0] / sw_per_sub)), dtype=np.float)
    pred = np.zeros(shape=(int(sw_pred.shape[0] / sw_per_sub)), dtype=np.float)
    for i in range(int(sw_label.shape[0] / sw_per_sub)):
        prob_y = np.mean(sw_label[sw_per_sub * i:sw_per_sub * (i + 1)])
        prob0 = np.mean(sw_pred[sw_per_sub * i:sw_per_sub * (i + 1), 0])
        prob1 = np.mean(sw_pred[sw_per_sub * i:sw_per_sub * (i + 1), 1])
        label[i] = prob_y
        prob[i] = prob1
        pred[i] = 0 if prob0 > prob1 else 1
    return label, prob, pred

def tensor_decision(sw_label, sw_pred, class_n, sw_per_sub):
    sw_label = torch.flatten(sw_label)
    sw_pred = torch.flatten(sw_pred)
    label = torch.zeros(int(sw_label.shape[0] / sw_per_sub))
    pred = torch.zeros(int(sw_pred.shape[0] / sw_per_sub))
    for i in range(0, int(sw_label.shape[0] / sw_per_sub)):
        counts1 = torch.bincount(sw_label[sw_per_sub * i:sw_per_sub * (i + 1)], minlength=class_n)
        counts2 = torch.bincount(sw_pred[sw_per_sub * i:sw_per_sub * (i + 1)], minlength=class_n)
        ind1 = torch.argmax(counts1)
        ind2 = torch.argmax(counts2)
        label[i] = ind1
        pred[i] = ind2
        # print(label, pred)
    return label, pred

def soft_tensor_decision(sw_label, sw_pred, class_n, sw_per_sub):
    sw_label = torch.flatten(sw_label)
    sw_pred = torch.flatten(sw_pred)
    label = torch.zeros(int(sw_label.shape[0] / sw_per_sub))
    pred = torch.zeros(int(sw_pred.shape[0] / sw_per_sub))
    for i in range(int(sw_label.shape[0] / sw_per_sub)):
        prob1 = torch.mean(sw_label[sw_per_sub * i:sw_per_sub * (i + 1)])
        prob2 = torch.mean(sw_pred[sw_per_sub * i:sw_per_sub * (i + 1)])
        label[i] = prob1
        pred[i] = prob2
    return label, pred

def log_csv(file, logs):
    f = open(file, mode='a', newline='')
    wr = csv.writer(f)
    wr.writerow([logs])
    f.close()

def log_txt(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()

def count_trainable_params(models):
    model_params = []
    for i in range(len(models)):
        model_params += list(models[i].parameters())
    tmp = filter(lambda x: x.requires_grad, model_params)
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    return num

def count_trainable_param(model):

    tmp = filter(lambda x: x.requires_grad, list(model.parameters()))
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    return num

