from typing import overload
import torch
from numpy.lib import save
from util import Logger, accuracy, TotalMeter
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data,roc_thr_plot,my_tpr_fpr
from util.loss import mixup_cluster_loss
import copy
from torch.optim import lr_scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



import os
from torch.autograd import Variable
from dataloader import BatchImageGenerator
from copy import deepcopy
from collections import OrderedDict
from typing import List
from datetime import datetime
from tensorboardX import SummaryWriter
from hua3dtu import huatu
import time
#import seaborn as sns
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
class BasicTrain:

    def __init__(self, train_config, model,repeat_time) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.epochs = train_config['epochs']
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.setup_path(repeat_time)
        self.weight_name = [name for name, _ in self.model.named_parameters()]
        self.weight_len = len(self.weight_name)
        #self.initialize_parameters()
        self.metainner_optim = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,weight_decay=0.0001
        )

        self.metaouter_optim = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,weight_decay=0.0001
        )
        self.innerscheduler = lr_scheduler.StepLR(self.metainner_optim, step_size=1000, gamma=0.5)
        self.outerscheduler = lr_scheduler.StepLR(self.metaouter_optim, step_size=1000, gamma=0.5)

        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,weight_decay=0.0001
        )
        self.optimscheduler = lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.5)
        #self.optimscheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min',factor=0.5,patience=300,verbose=True,min_lr=0.00001,threshold=0.0001, threshold_mode='rel')
        now = datetime.now()
        self.date_time = now.strftime("%m-%d-%H-%M-%S")
        self.writer = SummaryWriter(os.path.join('./logs/log_{}'.format(self.date_time)))
        self.group_loss = train_config['group_loss']

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']


        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy, \
        self.val_accuracy, self.test_accuracy, self.edges_num = [
            TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()



    def train(self):
        training_process = []
        global best_model_wts
        best_acc = 0
        scheduler = lr_scheduler.StepLR(self.optimizers[0], step_size=50, gamma=0.5)
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])

            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)
            scheduler.step()


            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Val AUC:{val_result[0]:.2f}',
            ]))

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result )
            if best_acc < val_result[0] and epoch > 1:
                best_acc = val_result[0]
                best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model_wts)

        test_result = self.test_per_epoch(self.test_dataloader,
                                          self.test_loss, self.test_accuracy)
        self.logger.info(" | ".join([
            f'Test Loss:{self.test_loss.avg: .3f}',
            f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
            f'Test AUC:{test_result[0]:.2f}'
        ]))
        training_process.append(test_result)
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)

    def setup_path(self,repeat_time):

        self.root_folder = 'D:/learning/23/23.7/FBNETGEN-main/'
        self.datas = ['YALE1.h5',
                      'USM1.h5',
                      'UM1.h5',
                      'UCLA1.h5',
                      'TRINITY1.h5',
                      'STANFORD1.h5',
                      'SDSU1.h5',
                      'SBL1.h5',
                      'PITT1.h5',
                      'OLIN1.h5',
                      'NYU1.h5',
                      'MAX_MUN1.h5',
                      'LEUVEN1.h5',
                      'KKI1.h5',
                      'CMU1.h5',
                      'CALTECH1.h5']

        self.paths = []

        for data in self.datas:
            path = os.path.join(self.root_folder, data)
            self.paths.append(path)

        if repeat_time==0:
          
            self.unseen_index = [2]
        elif repeat_time==1:
            self.unseen_index = [3]
        else:
            self.unseen_index = [10]



        #self.unseen_index = [3]

        self.unseen_data_paths=[]


        for k in self.unseen_index:
            path=os.path.join(self.root_folder, self.datas[k])
            self.unseen_data_paths.append(path)
            self.paths.remove(self.paths[k])
            self.datas.remove(self.datas[k])
        print(self.unseen_data_paths)

        self.val_index=[0,0]



        self.batImageGenTrains = []
        self.val_paths = []
        for k in self.val_index:
            path=os.path.join(self.root_folder, self.datas[k])
            self.val_paths.append(path)
            self.paths.remove(self.paths[k])
            self.datas.remove(self.datas[k])

        self.batImageGenVals = []
        for val_path in self.val_paths:
            batImageGenVal = BatchImageGenerator(file_path=val_path, stage='val',
                                                 b_unfold_label=False)
            self.batImageGenVals.append(batImageGenVal)

        for train_path in self.paths:
            batImageGenTrain = BatchImageGenerator(file_path=train_path, stage='train',
                                                   b_unfold_label=False)
            self.batImageGenTrains.append(batImageGenTrain)








    def sparsity_matrix_loss(self,matrix,label,len):
        mpos = torch.zeros(200, 200).to(device)

        mneg = torch.zeros(200, 200).to(device)
        labelpos=0
        labelneg=0
        for index,h in enumerate(label):
            if h==0:
                mpos=mpos+matrix[index]
                labelpos=labelpos+1
            else:
                mneg = mneg+matrix[index]
                labelneg = labelneg + 1
        if labelpos !=0:
            mpos = mpos / labelpos
        if labelneg !=0:
            mneg = mneg/labelneg

        overlap=abs(mpos-mneg)
        loss=torch.norm(overlap, p=1)/(200*200)
        return mpos,labelpos,mneg,labelneg,loss,overlap

    def overlap_loss(self,overlap1,overlap2):
        num=1
        overlap11=overlap1.view(num,-1)
        overlap22 = overlap2.view(num, -1)
        intersection = torch.sum(overlap11 * overlap22, dim=1)
        union = torch.sum(overlap11, dim=1) + torch.sum(overlap22, dim=1)
        dice_scores = 2 * intersection / (union + 1e-8)
        return 1 - dice_scores.mean()

    def matrix_entropy(self,matrix):
        matrix = matrix + 1e-10
        entropy = torch.sum(torch.abs(matrix))

        return entropy

    def entropy_loss(self,matrix):
        probabilities = F.softmax(matrix, dim=0)# 将矩阵的元素转换为概率分布
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=0)# 添加一个小值以避免log(0)的情况

        return entropy





    def train3(self):
        self.model.train()
        self.best_accuracy_val = -1
        inner_loops =5000
        torch.backends.cudnn.enabled = False

        for ite in range(inner_loops):
            outputs = []
            labels = []
            #idx_all = np.random.choice(len(self.batImageGenTrains), 13, replace=False)
            idx_all = np.random.choice(len(self.batImageGenTrains), 11, replace=False)
            train_loss = 0.0
            for idx in idx_all:
                final_fc_train, final_pearson_train, labels_train = self.batImageGenTrains[
                    idx].get_images_labels_batch_new()
                final_fc_train, final_pearson_train, labels_train = torch.from_numpy(
                    np.array(final_fc_train, dtype=np.float32)), torch.from_numpy(
                    np.array(final_pearson_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))
                labels_train = labels_train.long()
                inputs_train, pearson_train, labels_train = final_fc_train.to(
                    device), final_pearson_train.to(device), labels_train.to(device)
                inputs, nodes, targets_a, targets_b, lam = mixup_data(
                    inputs_train, pearson_train, labels_train, -1, device)
                outputs_train, _, _ = self.model(inputs, nodes)
                outputs.append(outputs_train)
                labels.append(labels_train)
                loss = 2 * mixup_criterion(
                    self.loss_fn, outputs_train, targets_a, targets_b, lam)
                train_loss = train_loss + loss

            # m = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5],outputs[6], outputs[7], outputs[8], outputs[9], outputs[10], outputs[11], outputs[12]], 0)
            # label_val = torch.cat([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],labels[6], labels[7], labels[8], labels[9], labels[10], labels[11], labels[12]], 0)
            m = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5],outputs[6], outputs[7], outputs[8], outputs[9], outputs[10]], 0)
            label_val = torch.cat([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],labels[6], labels[7], labels[8], labels[9], labels[10]],0)
            acc = accuracy(m, label_val)[0]
            #print(ite,acc)
            #print(train_loss)
            self.optim.zero_grad()
            #self.model.train()
            train_loss.backward()
            self.optim.step()
            self.optimscheduler.step()
            #flags_log = os.path.join('newlogs/', 'baseline' + self.date_time + 'read_log.txt') #这个是大修设的，实际上应该用下面的
            flags_log = os.path.join('newlogs/', 'baseline'+self.date_time + 'read_log.txt')
            f = open(flags_log, mode='a')
            f.write(str(ite))
            f.write('train accuracy:')
            f.write(str(acc))
            f.close()

            val_acc = self.test_workflow(self.batImageGenVals,ite)
            f = open(flags_log, mode='a')
            f.write(str(ite))
            f.write('val accuracy:')
            f.write(str(val_acc))
            f.write('\n')
            f.close()
            self.writer.add_scalars('Acc', {'train_acc': acc, 'val_acc': val_acc}, ite)

            # if ite % 2 == 0 and ite is not 0 :
            #     self.test_workflow(self.batImageGenVals)



    def train4(self):
        self.model.train()

        self.best_accuracy_val = -1
        self.model.to(device)
        inner_loops =5000
        #self.model.load_state_dict(torch.load('models/'+'01-16-15-45-01'+'best.mdl'))



        for ite in range(inner_loops):

            self.model.train()
            # torch.save(self.model.state_dict(), 'models/' + 'now.mdl')
            #modelparms = deepcopy(self.model.state_dict())
            # select the validation domain for meta val
            meta_idx_all = np.random.choice(len(self.batImageGenTrains), 13, replace=False)#########################正版是13
            meta_train_loss = 0.0
            for i in range(6):
                final_fc_train, final_pearson_train, labels_train = self.batImageGenTrains[
                    meta_idx_all[i]].get_images_labels_batch()
                final_fc_train, final_pearson_train, labels_train = torch.from_numpy(
                    np.array(final_fc_train, dtype=np.float32)), torch.from_numpy(
                    np.array(final_pearson_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))
                labels_train = labels_train.long()
                inputs_train, pearson_train, labels_train = final_fc_train.to(
                    device), final_pearson_train.to(device), labels_train.to(device)
                inputs, nodes, targets_a, targets_b, lam = mixup_data(
                    inputs_train, pearson_train, labels_train, 1, device)
                #outputs_train, _, _ = self.model(inputs_train, pearson_train)
                outputs_train, _, _ = self.model(inputs, nodes)
                torch.save(self.model.state_dict(), 'models/' + 'now.mdl')
                #loss = F.nll_loss(torch.log(outputs_train), labels_train)
                loss = 2 * mixup_criterion(
                    self.loss_fn, outputs_train, targets_a, targets_b, lam)
                meta_train_loss = meta_train_loss + loss

            self.metainner_optim.zero_grad()
            #meta_train_loss.backward()
            meta_train_loss.backward(retain_graph=True)
            self.metainner_optim.step()
            #self.innerscheduler.step(meta_train_loss)
            #self.innerscheduler.step()
            #meta_idx_val=np.delete(meta_idx_all,[0,1,2,3])

            meta_idx_val=np.delete(meta_idx_all,[0,1,2,3,4,5])####################这个才是正版


            meta_val_loss=0.0
            overlap_all=[]
            outputs=[]
            labels=[]
            meta_val_loss_all=[]

            for k in meta_idx_val:
                batImageMetaVal=self.batImageGenTrains[k]
                final_fc_val, final_pearson_val, labels_val = batImageMetaVal.get_images_labels_batch()
                final_fc_val, final_pearson_val, labels_val = torch.from_numpy(
                    np.array(final_fc_val, dtype=np.float32)), torch.from_numpy(
                    np.array(final_pearson_val, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_val, dtype=np.float32))
                labels_val = labels_val.long()
                inputs_val, pearson_val, labels_val = final_fc_val.to(
                    device), final_pearson_val.to(device), labels_val.to(device)
                inputs, nodes, targets_a, targets_b, lam = mixup_data(
                    inputs_val, pearson_val, labels_val, 1, device)
                #outputs_val, learnable_matrix, edge_variance = self.model(inputs_val, pearson_val)
                outputs_val, learnable_matrix, edge_variance = self.model(inputs, nodes)
                outputs.append(outputs_val)
                labels.append(labels_val)
                croloss = 2 * mixup_criterion(
                    self.loss_fn, outputs_val, targets_a, targets_b, lam)
                #croloss=F.nll_loss(torch.log(outputs_val), labels_val)
                #meta_val_loss_all.append(croloss)
                meta_val_loss = meta_val_loss + croloss



                # if self.group_loss:
                #     meta_val_loss = meta_val_loss +  mixup_cluster_loss(learnable_matrix,
                #                                                                targets_a, targets_b, lam)
                if self.sparsity_loss:
                    mpos, labelpos, mneg, labelneg, sparsity_loss, overlap = self.sparsity_matrix_loss(
                        learnable_matrix, labels_val, len(labels_val))
                    # sparsity_loss1 = 1.0e-5* torch.norm(learnable_matrix, p=1)
                    # meta_val_loss = meta_val_loss + sparsity_loss1

                    if labelpos != 0 and labelneg != 0:

                        #meta_val_loss = meta_val_loss + sparsity_loss * 1
                        mean = torch.mean(overlap, dim=0)
                        std = torch.std(overlap, dim=0)
                        overlap_normalized = (overlap - mean) / std
                        sigmoid_matrix = torch.sigmoid(overlap_normalized)


                        entropyloss=self.matrix_entropy(sigmoid_matrix)

                        meta_val_loss = meta_val_loss + entropyloss * 0.01#sp
                        overlap_all.append(sigmoid_matrix)

            # meta_val_loss_all = torch.tensor(meta_val_loss_all)
            # variance = torch.var(meta_val_loss_all)
            # meta_val_loss=meta_val_loss+variance*10


            h=0
            for k in range(len(overlap_all)-1):
                for i in range(len(overlap_all)-k-1):
                    meta_val_loss = meta_val_loss + self.overlap_loss(overlap_all[i+k+1], overlap_all[k]) * 0.01#cons
                    h=h+1


            m = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6]], 0)
            label_val = torch.cat([labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6]], 0) ##################这才是正版

            # m = torch.cat([outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]], 0)
            # label_val = torch.cat([labels[0], labels[1], labels[2], labels[3], labels[4]], 0)
            acc = accuracy(m, label_val)[0]

            #print('train acc: ', acc)
            # init the grad to zeros first
            self.metaouter_optim.zero_grad()
            #torch.autograd.set_detect_anomaly(True)

            # backward your network
            meta_val_loss.backward()
            self.model.load_state_dict(torch.load('models/' + 'now.mdl'))
            self.model.train()
            #self.model.load_state_dict(modelparms)
            # optimize the parameters
            self.metaouter_optim.step()
            #self.outerscheduler.step(meta_val_loss)
            #self.outerscheduler.step()
            #print(ite, meta_val_loss)

            flags_log = os.path.join('newlogs/dyn/', 'metalearning_GraphSAGE'+self.date_time+'read_log.txt')#这个是大修设的，实际上应该用下面的
            #flags_log = os.path.join('logs/', 'metalearning'+self.date_time+'read_log.txt')
            f = open(flags_log, mode='a')
            f.write(str(ite))
            f.write('train accuracy:')
            f.write(str(acc))
            f.close()
            val_acc=self.test_workflow(self.batImageGenVals,ite)
            self.writer.add_scalars('Acc', {'train_acc': acc,'val_acc':val_acc}, ite)
            f = open(flags_log, mode='a')
            f.write('val accuracy:')
            f.write(str(val_acc))
            f.write('\n')
            f.close()





    def write_log(self,log1, log2,log_path):
        f = open(log_path, mode='a')
        f.write('ite:')
        f.write(str(log1))
        f.write('accuracy:')
        f.write(str(log2))
        f.write('\n')
        f.close()

    def write_log2(self,log,log_path):
        f = open(log_path, mode='a')
        f.write('all_accuracy:')
        f.write(str(log))
        f.write('\n')
        f.close()

    def write_log3(self,log,log_path):
        f = open(log_path, mode='a')
        f.write('all_auc:')
        f.write(str(log))
        f.write('\n')
        f.close()


    def Matrixdifference(self,matrix,pearson_matrix,label):
        mpos = np.zeros((200, 200))
        mneg = np.zeros((200, 200))
        labelpos=0
        labelneg=0

        ppos = np.zeros((200, 200))
        pneg = np.zeros((200, 200))
        index0 = np.where(label == 0)
        index1=np.where(label==1)



        for index,h in enumerate(label):
            if h==0:
                mpos=mpos+matrix[index]
                ppos=ppos+pearson_matrix[index]
                labelpos=labelpos+1
            else:
                mneg = mneg+matrix[index]
                pneg= pneg+pearson_matrix[index]
                labelneg = labelneg + 1
        if labelpos !=0:
            mpos = mpos / labelpos
            ppos= ppos / labelpos
        if labelneg !=0:
            mneg = mneg/labelneg
            pneg = pneg / labelneg

        max_value = np.max(mpos)
        min_value = np.min(mpos)
        range_value = max_value - min_value
        mpos_matrix = (mpos - min_value) / range_value
        huatu(mpos_matrix, 'mpos_matrix')
        np.save('D:/learning/23/23.7/FBNETGEN-main/hua/NYU_FB/02-22-11-25-05mpos_matrix', mpos_matrix,
                allow_pickle=True, fix_imports=True)
        max_value = np.max(mneg)
        min_value = np.min(mneg)
        range_value = max_value - min_value
        mneg_matrix = (mneg - min_value) / range_value
        huatu(mneg_matrix, 'mneg_matrix')
        np.save('D:/learning/23/23.7/FBNETGEN-main/hua/NYU_FB/02-22-11-25-05mneg_matrix', mneg_matrix,
                allow_pickle=True, fix_imports=True)
        max_value = np.max(ppos)
        min_value = np.min(ppos)
        range_value = max_value - min_value
        ppos_matrix = (ppos - min_value) / range_value
        huatu(ppos_matrix, 'ppos_matrix')
        np.save('D:/learning/23/23.7/FBNETGEN-main/hua/NYU_FB/02-22-11-25-05ppos_matrix', ppos_matrix,
                allow_pickle=True, fix_imports=True)
        max_value = np.max(pneg)
        min_value = np.min(pneg)
        range_value = max_value - min_value
        pneg_matrix = (pneg - min_value) / range_value
        huatu(pneg_matrix, 'pneg_matrix')
        np.save('D:/learning/23/23.7/FBNETGEN-main/hua/NYU_FB/02-22-11-25-05pneg_matrix', pneg_matrix,
                allow_pickle=True, fix_imports=True)

        overlap = mpos - mneg

        overlap=abs(mpos-mneg)
        max_value = np.max(overlap)
        min_value = np.min(overlap)
        range_value = max_value - min_value
        scaled_matrix = (overlap - min_value) / range_value
        #keshihua(scaled_matrix)
        huatu(scaled_matrix, 'inter_learnable_matrix')
        np.save('D:/learning/23/23.7/FBNETGEN-main/hua/NYU_FB/02-22-11-25-05inter_learnable_matrix', scaled_matrix, allow_pickle=True, fix_imports=True)

        overlap_pearson = abs(ppos - pneg)
        max_value = np.max(overlap_pearson)
        min_value = np.min(overlap_pearson)
        range_value = max_value - min_value
        scaled_matrix = ( overlap_pearson - min_value) / range_value
        huatu(scaled_matrix, 'inter_pearson_matrix')
        np.save('D:/learning/23/23.7/FBNETGEN-main/hua/NYU_FB/02-22-11-25-05inter_pearson_matrix', scaled_matrix, allow_pickle=True,
                fix_imports=True)

        # plt.imshow(scaled_matrix, cmap='hot', interpolation='nearest')
        # plt.colorbar()
        # plt.savefig('roc_new/' + 'metalearning' + self.date_time  + '.png')
        # plt.clf()

        # 显示图像
        #plt.show()

        return overlap

    def heldout_test(self):


        self.model.load_state_dict(torch.load('models/' + self.date_time + 'best.mdl'))


        self.batImageGenTests=[]
        self.model.eval()
        localtime = time.localtime(time.time())
        timestring = time.strftime('%Y/%m/%d - %H:%M:%S')
        flags_log = os.path.join('newlogs/dyn/', 'metalearning_GraphSAGE' + self.date_time + 'test_log.txt')#这个是大修设的，实际上应该用下面的
        #flags_log = os.path.join('logs/', 'test_log.txt')
        #flags_log = os.path.join('logs/', 'con_loss_0.01_sp_0.0001_test_log.txt')
        self.write_log2(timestring, flags_log)
        for unseen_data_path in self.unseen_data_paths:
            batImageGenTest = BatchImageGenerator( file_path=unseen_data_path, stage='test',
                                                 b_unfold_label=False)
            self.batImageGenTests.append(batImageGenTest)

        accuracies = []
        results = []
        labels = []
        matrixs=[]
        pearson_matrixs=[]
        lengths = 0
        all_accuracies = 0
        for count, batImageGenTest in enumerate(self.batImageGenTests):
            accuracy_val,result,label,m,pearson_matrix= self.testout(batImageGenTest=batImageGenTest)
            print('ite:',count, 'accuracy:',accuracy_val)
            self.write_log(count,accuracy_val, flags_log)
            accuracies.append(accuracy_val)
            length = len(batImageGenTest.labels)
            all_accuracies = accuracy_val * length + all_accuracies
            lengths = length + lengths
            matrixs.extend(m)
            results.extend(result)
            labels.extend(label)
            pearson_matrixs.extend(pearson_matrix)
            #print(2)

        labels=np.array(labels)
        results = np.array(results)
        matrixs=np.array(matrixs)
        pearson_matrixs=np.array(pearson_matrixs)
        #self.Matrixdifference(matrixs,pearson_matrixs,labels)

        mean_acc = all_accuracies / lengths
        auc = roc_auc_score(labels, results)
        # fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, results)
        # plt.plot(fpr, tpr)
        # plt.show()
        results = np.array(results)
        #np.save('D:/learning/23/23.7/FBNETGEN-main/t_test/du_UCLA_results.npy', results)

        results[results > 0.5] = 1
        results[results <= 0.5] = 0
        precision, recall, f1_score, support = precision_recall_fscore_support(
            labels, results, average=None)
        f = open(flags_log, mode='a')
        f.write('precision:')
        f.write(str(precision))
        f.write('\n')
        f.write('recall:')
        f.write(str(recall))
        f.write('\n')
        f.write('f1_score:')
        f.write(str(f1_score))
        f.write('\n')
        f.write('support:')
        f.write(str(support))
        f.write('\n')
        f.close()



        #mean_acc = np.mean(accuracies)
        print('----------accuracy test final----------:', mean_acc)
        print('----------auc test final----------:',auc)
        #self.write_log2(mean_acc, flags_log)
        self.write_log3(auc, flags_log)





    def testout(self, batImageGenTest):
        final_fc_test = batImageGenTest.final_fc
        final_pearson_test = batImageGenTest.final_pearson
        labels_test = batImageGenTest.labels
        average_matrix = np.mean(final_pearson_test, axis=0)
        # np.save('D:/learning/23/23.7/FBNETGEN-main/matrix/pearson_matrix',average_matrix, allow_pickle=True,
        #         fix_imports=True)
        threshold = 10
        result=[]
        labels=[]
        if len(labels_test) > threshold:
            n_slices_test = len(labels_test) // threshold
            indices_test = []

            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(labels_test) * (per_slice + 1) // n_slices_test)
            test_final_fc_splits = np.split(final_fc_test, indices_or_sections=indices_test)
            test_final_pearson_splits = np.split(final_pearson_test, indices_or_sections=indices_test)

            test_final_fc_splits_2_whole = np.concatenate(test_final_fc_splits)
            test_final_pearson_splits_2_whole = np.concatenate(test_final_pearson_splits)
            assert np.all(final_fc_test == test_final_fc_splits_2_whole), np.all(
                final_pearson_test == test_final_pearson_splits_2_whole)
            labels_preds = []
            matrix=[]
            pearson_matrix=[]
            for test_final_fc_split, test_final_pearson_split in zip(test_final_fc_splits,
                                                                     test_final_pearson_splits):
                final_fc_test, final_pearson_test = torch.from_numpy(
                    np.array(test_final_fc_split, dtype=np.float32)), torch.from_numpy(
                    np.array(test_final_pearson_split, dtype=np.float32))
                inputs_test, pearson_test = final_fc_test.to(
                    device), final_pearson_test.to(device)
                output, m, _ = self.model(inputs_test, pearson_test)
                # numpy_matrix = m[1].cpu().detach().numpy()
                # plt.imshow(numpy_matrix, cmap='hot', interpolation='nearest')
                # plt.colorbar()
                #
                # # 显示图像
                # plt.show()

                output = output.cpu().data.numpy()
                m=m.cpu().data.numpy()
                pearson_test=pearson_test.cpu().data.numpy()
                pearson_matrix.append(pearson_test)
                matrix.append(m)
                labels_preds.append(output)
            matrixs= np.concatenate(matrix)
            pearson_matrix=np.concatenate(pearson_matrix)
            predictions = np.concatenate(labels_preds)
            labels_test = torch.from_numpy(
                np.array(labels_test, dtype=np.float32))
            labels_test = labels_test.long()
            predictions = torch.from_numpy(predictions)
            top1 = accuracy(predictions, labels_test)[0]
            #result +=predictions[:, 1].tolist()
            result += F.softmax(predictions, dim=1)[:, 1].tolist()
            labels += labels_test.tolist()
            return top1,result,labels,matrixs,pearson_matrix

    #从mldg抄过来的

    def test_workflow(self, batImageGenVals,ite):

        accuracies = []
        results=[]
        labels=[]
        predictions=[]

        lengths=0
        all_accuracies=0
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val,result,label,prediction = self.test2(batImageGenTest=batImageGenVal)
            accuracies.append(accuracy_val)
            results.extend(result)
            labels.extend(label)
            predictions.extend(prediction)
            length=len(batImageGenVal.labels)
            all_accuracies=accuracy_val*length+all_accuracies
            lengths=length+lengths
        # flags_log = os.path.join('newlogs/dyn/', 'metalearning_GraphSAGE' + self.date_time + 'val_log.txt')#这个是大修设的，实际上应该用下面的
        # #flags_log = os.path.join('logs/', 'metalearning' + self.date_time + 'val_log.txt')
        # f = open(flags_log, mode='a')
        # f.write('\n')
        # h=[0,10,20,30,40,50]
        # for k in h:
        #     f.write('output:')
        #     f.write(str(predictions[k]))
        #     f.write(' ')
        #     f.write('result:')
        #     f.write(str(results[k]))
        #     f.write(' ')
        #     f.write('label:')
        #     f.write(str(labels[k]))
        #     f.write('\n')
        #
        # f.close()
        mean_acc=all_accuracies/lengths

        #auc = roc_auc_score(labels, results)


        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc
            torch.save(self.model.state_dict(),'newmodels/'+self.date_time+'best_GraphSAGE.mdl')#这个是大修设的，实际上应该用下面的
            #torch.save(self.model.state_dict(), 'models/' + self.date_time + 'best.mdl')
        return mean_acc


    def test2(self,batImageGenTest):
        # switch on the network test mode
        labels = []
        result = []
        self.model.eval()
        final_fc_test=batImageGenTest.final_fc
        final_pearson_test=batImageGenTest.final_pearson
        labels_test = batImageGenTest.labels
        threshold = 10
        if len(labels_test) > threshold:
            n_slices_test = len(labels_test) // threshold
            indices_test = []

            for per_slice in range(n_slices_test - 1):
                indices_test.append(len(labels_test) * (per_slice + 1) // n_slices_test)
            test_final_fc_splits = np.split(final_fc_test, indices_or_sections=indices_test)
            test_final_pearson_splits = np.split(final_pearson_test, indices_or_sections=indices_test)

            test_final_fc_splits_2_whole = np.concatenate(test_final_fc_splits)
            test_final_pearson_splits_2_whole = np.concatenate(test_final_pearson_splits)
            assert np.all(final_fc_test == test_final_fc_splits_2_whole),np.all(final_pearson_test == test_final_pearson_splits_2_whole)
            labels_preds = []
            for test_final_fc_split,test_final_pearson_split in zip(test_final_fc_splits,test_final_pearson_splits):
                final_fc_test, final_pearson_test = torch.from_numpy(
                    np.array(test_final_fc_split, dtype=np.float32)), torch.from_numpy(
                    np.array(test_final_pearson_split, dtype=np.float32))
                inputs_test, pearson_test = final_fc_test.to(
                    device), final_pearson_test.to(device)
                output, _, _ = self.model(inputs_test, pearson_test)
                output=output.cpu().data.numpy()
                labels_preds.append(output)
            predictions = np.concatenate(labels_preds)


            labels_test=torch.from_numpy(
            np.array(labels_test, dtype=np.float32))
            labels_test = labels_test.long()
            predictions=torch.from_numpy(predictions)
            #labels_test=labels_test.to(device)
            top1 = accuracy(predictions, labels_test)[0]
            result += F.softmax(predictions, dim=1)[:, 1].tolist()
            labels += labels_test.tolist()
            #auc = roc_auc_score(labels, result)

            return top1,result,labels,predictions


