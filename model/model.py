from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
import math
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
import itertools



def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = sample_gumbel(logits.size(), eps=eps)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape).cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


class GruKRegion(nn.Module):

    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(kernel_size, kernel_size, layers,
                       bidirectional=True, batch_first=True)

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size*2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size)
        )

    def forward(self, raw):

        b, k, d = raw.shape

        x = raw.view((b*k, -1, self.kernel_size))

        x, h = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)
        return x


class ConvKRegion(nn.Module):

    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(in_channels=k, out_channels=32,
                            kernel_size=kernel_size, stride=2)

        output_dim_1 = (time_series - kernel_size) // 2 + 1

        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16,
                            kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size)
        )

    def forward(self, x):

        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b*k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class SeqenceModel(nn.Module):

    def __init__(self, model_config, roi_num=360, time_series=512):
        super().__init__()

        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series, pool_size=4, )
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'], dropout=model_config['dropout'])

        self.linear = nn.Sequential(
            Linear(model_config['embedding_size']*roi_num, 256),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(256, 32),
            nn.Dropout(model_config['dropout']),
            nn.ReLU(),
            Linear(32, 2)
        )

    def forward(self, x):
        x = self.extract(x)
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        return x


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):

        m = torch.einsum('ijk,ipk->ijp', x, x)

        m = torch.unsqueeze(m, -1)

        return m


class MHSA(nn.Module):
    def __init__(self, dim, roi_num=264):
        super().__init__()

        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.num_heads = 4            #4

    def forward(self, x):
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)

        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        m = torch.einsum('ijk,ipk->ijp', v, v)

        m = torch.unsqueeze(m, -1)

        return m


class Embed2GraphByTans(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()
        self.transformer = nn.Transformer(d_model=input_dim)
        self.fc = nn.Linear(input_dim, 200)

    def forward(self, x):
        output=self.transformer(x,x)
        x=self.fc(output)
        m = torch.einsum('ijk,ipk->ijp', x, x)
        m = torch.unsqueeze(m, -1)
        return m

class Embed2GraphByLinear(nn.Module):

    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                            enumerate(classes)}
            labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                     dtype=np.int32)
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):

        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(
            x, (batch_sz, region_num, region_num, -1))
        return m


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(in_dim, out_dim * heads)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, adj, x):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, -1)

        # 计算注意力系数
        h_cat = torch.cat([h.unsqueeze(2).expand(-1, -1, N, -1, -1),
                           h.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=-1)
        e = self.attn(h_cat).squeeze(-1)
        mask = (adj.unsqueeze(-1) * -1e9)
        e = F.leaky_relu(e + mask)
        a = F.softmax(e, dim=2)

        # 多头聚合
        out = torch.einsum('bijh,bjhd->bihd', a, h)
        return out.reshape(B, N, -1)


class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, adj, x):
        # 邻居均值聚合
        neighbor_mean = torch.matmul(adj, x) / (adj.sum(-1, keepdim=True) + 1e-9)
        combined = torch.cat([x, neighbor_mean], -1)
        return F.leaky_relu(self.linear(combined))


class GNNWrapper(nn.Module):
    def __init__(self, model_type, node_input_dim, roi_num=200):
        super().__init__()
        self.model_type = model_type

        # 公共参数
        self.inner_dim = 64
        self.final_dim = 8

        # GAT配置
        if model_type == "GAT":
            self.gnn_layers = nn.ModuleList([
                GATLayer(node_input_dim, self.inner_dim, heads=2),
                GATLayer(self.inner_dim * 2, self.inner_dim, heads=2),
                GATLayer(self.inner_dim * 2, self.final_dim, heads=2)
            ])

        # GraphSAGE配置
        else :
            self.gnn_layers = nn.ModuleList([
                SAGELayer(node_input_dim, self.inner_dim),
                SAGELayer(self.inner_dim, self.inner_dim),
                SAGELayer(self.inner_dim, self.final_dim)
            ])

        self.fcn = nn.Sequential(
            #nn.Linear(8 * roi_num, 256),
            nn.Linear(3200, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2),
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, adj, x):
        bz=x[0]
        for layer in self.gnn_layers:
            x = layer(adj, x)
        #x = x.view(bz, -1)
        x = x.flatten(1)
        return self.fcn(x)







class GNNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)#8的位置本来是inner_dim
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)#8的位置本来是inner_dim

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        ###################自己添加的###########################
        # self.gcn3 = nn.Sequential(
        #     nn.Linear(inner_dim, inner_dim),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        # self.bn4 = torch.nn.BatchNorm1d(inner_dim)
        # self.gcn4 = nn.Sequential(
        #     nn.Linear(inner_dim, inner_dim),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        # self.bn5 = torch.nn.BatchNorm1d(inner_dim)
        ###################自己添加的###########################
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dropout = nn.Dropout(0.5)
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8*roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2),
            nn.Sigmoid()
            #nn.Softmax(dim=1)
        )


    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum('ijk,ijp->ijp', m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn1(x)

        x = x.reshape((bz*self.roi_num, -1))
        x = self.bn2(x)
###################自己添加的###########################
        # x = x.reshape((bz, self.roi_num, -1))
        # x = torch.einsum('ijk,ijp->ijp', m, x)
        # x = self.gcn3(x)
        # x = x.reshape((bz * self.roi_num, -1))
        # x = self.bn4(x)
        #
        # x = x.reshape((bz, self.roi_num, -1))
        # x = torch.einsum('ijk,ijp->ijp', m, x)
        # x = self.gcn4(x)
        # x = x.reshape((bz * self.roi_num, -1))
        # x = self.bn5(x)
        ###################################


        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum('ijk,ijp->ijp', m, x)

        x = self.gcn2(x)
        x=self.dropout(x)

        x = self.bn3(x)

        x = x.view(bz,-1)

        x = self.fcn(x)

        return x


class FBNETGEN(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        self.graph_generation = model_config['graph_generation']
        if model_config['extractor_type'] == 'cnn':
            self.extract = ConvKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                time_series=time_series)
        elif model_config['extractor_type'] == 'gru':
            self.extract = GruKRegion(
                out_size=model_config['embedding_size'], kernel_size=model_config['window_size'],
                layers=model_config['num_gru_layers'])
        if self.graph_generation == "linear":
            # self.emb2graph = MHSA(
            #     model_config['embedding_size'], roi_num=roi_num)#####这个才是正版
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)
            # self.emb2graph = Embed2GraphByLinear(
            #     model_config['embedding_size'], roi_num=roi_num)
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_config['embedding_size'], roi_num=roi_num)

        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)####这个才是正版
        # self.predictor = GNNWrapper(
        #     "GAT",  # 从配置中选择GNN类型
        #     200,
        #     200
        # )

    def forward(self, t, nodes):
        x = self.extract(t)
        x = F.softmax(x, dim=-1)
        m = self.emb2graph(x)
        # p=torch.randn_like(m)
        # m = m + torch.randn_like(m) * 0.4
        # nodes= nodes + torch.randn_like(nodes) * 0.4
        #m = F.softmax(m, dim=-1)


        m = m[:, :, :, 0]

        bz, _, _ = m.shape

        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))
        out=self.predictor(m, nodes)

        return out, m, edge_variance


class HU_FBNETGEN(nn.Module):

    def __init__(self, model_config, roi_num=360, node_feature_dim=360, time_series=512):
        super().__init__()
        self.roi_num = roi_num
        self.node_feature_dim = node_feature_dim
        self.predictor = GNNPredictor(node_feature_dim, roi_num=roi_num)



    def forward(self, t, nodes):
        # t: (batch_size, n_roi, time_points)
        # nodes: (batch_size, n_roi, node_feature_dim)

        # 计算互信息矩阵
        m = t
        bz, _, _ = m.shape


        # 将互信息矩阵作为邻接矩阵传递给 GNN
        out = self.predictor(m, nodes)

        return out, m, torch.mean(torch.var(m.reshape((m.shape[0], -1)), dim=1))


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(torch.nn.Module):
    def __init__(self, roi_num):
        super().__init__()
        self.in_planes = 1
        self.d = roi_num

        self.e2econv1 = E2EBlock(1, 16, roi_num, bias=True)
        self.e2econv2 = E2EBlock(16, 32, roi_num, bias=True)
        self.E2N = torch.nn.Conv2d(32, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 32, (self.d, 1))
        self.dense1 = torch.nn.Linear(32, 2)
        # self.dense2 = torch.nn.Linear(8, 4)
        # self.dense3 = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        # out = F.dropout(F.leaky_relu(
        #     self.dense2(out), negative_slope=0.33), p=0.5)
        # out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out


class FCNet(nn.Module):

    def __init__(self, node_size, seq_len, kernel_size=3):
        super().__init__()

        self.ind1, self.ind2 = torch.triu_indices(node_size, node_size, offset=1)

        seq_len -= kernel_size//2*2
        channel1 = 32
        self.block1 = nn.Sequential(
            Conv1d(in_channels=1, out_channels=channel1,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel1),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2  

        seq_len -= kernel_size//2*2
        channel2 = 64
        self.block2 = nn.Sequential(
            Conv1d(in_channels=channel1, out_channels=channel2,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        seq_len //= 2 

        seq_len -= kernel_size//2*2
        channel3 = 96
        self.block3 = nn.Sequential(
            Conv1d(in_channels=channel2, out_channels=channel3,
                            kernel_size=kernel_size),
            nn.BatchNorm1d(channel3),
            nn.LeakyReLU()
        )

        channel4 = 64
        self.block4 = nn.Sequential(
            Conv1d(in_channels=channel3, out_channels=channel4,
                            kernel_size=kernel_size),
            Conv1d(in_channels=channel4, out_channels=channel4,
                            kernel_size=kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2)  
        )
        seq_len -= kernel_size//2*2
        seq_len -= kernel_size//2*2
        seq_len //= 2  
        
               
        self.fc = nn.Linear(in_features=seq_len*channel4, out_features=32)

        self.diff_mode = nn.Sequential(
            nn.Linear(in_features=32*200, out_features=32),
            nn.Linear(in_features=32, out_features=32),
            nn.Linear(in_features=32, out_features=2)
        )

    def forward(self, x):
        bz, _, time_series = x.shape

        #x = x.reshape((bz*2, 1, time_series))
        x = x.reshape((-1, 1, time_series))


        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        
        #x = x.reshape((bz, 2, -1))
        x = x.reshape((bz, 200, -1))

        x = self.fc(x)

        x = x.reshape((bz, -1))

        diff = self.diff_mode(x)

        return diff

        
