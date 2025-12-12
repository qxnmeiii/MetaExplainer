import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
def mixup_data(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = lam * nodes + (1 - lam) * nodes[index, :]
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam


def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    mix_xs, mix_nodes, mix_ys = [], [], []

    for t_y in y.unique():
        idx = y == t_y

        t_mixed_x, t_mixed_nodes, _, _, _ = mixup_data(
            x[idx], nodes[idx], y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        mix_nodes.append(t_mixed_nodes)

        mix_ys.append(y[idx])

    return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cal_step_connect(connectity, step):
    multi_step = connectity
    for _ in range(step):
        multi_step = np.dot(multi_step, connectity)
    multi_step[multi_step > 0] = 1
    return multi_step


def obtain_partition(dataloader, fc_threshold, step=2):
    pearsons = []
    for data_in, pearson, label in dataloader:
        pearsons.append(pearson)

    fc_data = torch.mean(torch.cat(pearsons), dim=0)

    fc_data[fc_data > fc_threshold] = 1
    fc_data[fc_data <= fc_threshold] = 0

    _, n = fc_data.shape

    final_partition = torch.zeros((n, (n-1)*n//2))

    connection = cal_step_connect(fc_data, step)
    temp = 0
    for i in range(connection.shape[0]):
        temp += i
        for j in range(i):
            if connection[i, j] > 0:
                final_partition[i, temp-i+j] = 1
                final_partition[j, temp-i+j] = 1
                # a = random.randint(0, n-1)
                # b = random.randint(0, n-1)
                # final_partition[a, temp-i+j] = 1
                # final_partition[b, temp-i+j] = 1

    connect_num = torch.sum(final_partition > 0)/n
    print(f'Final Partition {connect_num}')

    return final_partition.cuda().float(), connect_num


def stat_tfpr(data_true, data_pred, thr, tol=6):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i, j in zip(data_true, data_pred):
        if j >= thr:  # pred: positive
            # pred = 1
            if i == 1:  # real positive
                TP += 1
            else:  # real negative
                FP += 1
        else:
            # pred = 0
            if i == 0:  # real negative
                TN += 1
            else:  # real positive
                FN += 1
    # 真阳率：TPR = TP/(TP+FN)
    # 假阳率：FPR = FP/(FP+TN)
    tpr = round(float(TP) / float(TP + FN), tol)
    fpr = round(float(FP) / float(FP + TN), tol)
    return tpr, fpr


def my_range(minx, maxx, n, tol=6):
    span = maxx - minx
    step = round(float(span) / n, tol)
    # print(span, step, minx)
    x = minx
    for i in range(n):
        x += step
        yield round(x, tol)


def my_tpr_fpr(data_true, data_pred, n, tol=6):
    max_prd = max(data_pred)
    min_prd = min(data_pred)
    step = float(max_prd - min_prd) / n
    tpr_lst, fpr_lst, thr_lst = [], [], []
    result_lst = []
    for i in my_range(min_prd, max_prd, n, tol):
        tpr, fpr = stat_tfpr(data_true, data_pred, i, tol)
        tpr_lst.append(tpr)
        fpr_lst.append(fpr)
        thr_lst.append(i)
    return tpr_lst, fpr_lst, thr_lst


def all_max_idx(lst):
    mx = max(lst)
    idx_lst = []
    for i, j in enumerate(lst):
        if j == mx:
            idx_lst.append(i)
    return idx_lst


def find_optimal_cutoff_lst(TPR, FPR, threshold):
    # y = TPR - FPR
    # Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    yd_lst = [round(i, 6) for i in np.array(TPR) - np.array(FPR)]
    mx_idxs = all_max_idx(yd_lst)
    lst = []
    for Youden_index in mx_idxs:
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        lst.append([optimal_threshold, point])
    return lst


def roc_thr_plot(fpr, tpr, thrs,ite,date_time):
    roc_auc = metrics.auc(fpr, tpr)
    #plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, label='Val AUC = %0.3f' % roc_auc)  # 'b',
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1])  # , 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])

    # optimal_th, optimal_point = find_optimal_cutoff(TPR=tpr, FPR=fpr, threshold=thrs)
    optimal_lst = find_optimal_cutoff_lst(TPR=tpr, FPR=fpr, threshold=thrs)
    for i, optimal_v in enumerate(optimal_lst):
        optimal_th, optimal_point = optimal_v
        plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
        plt.text(optimal_point[0] + 0.02 * (i + 1), optimal_point[1] - 0.08 * (i + 1),
                 'Threshold:{optimal_th:.2f} [{a:.4f}, {b:.4f}]'
                 ''.format(optimal_th=optimal_th, a=optimal_point[0], b=optimal_point[1]))

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_new/' + 'metalearning' + date_time + str(ite) + '.png')
    plt.clf()
    #plt.show()

