from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from network.ChebConv import ChebConv, _ResChebGC


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


gan_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
                          [0, 5], [5, 6], [6, 7], [7, 8],
                          [0, 9], [9, 10], [10, 11], [11, 12],
                          [0, 13], [13, 14], [14, 15], [15, 16],
                          [0, 17], [17, 18], [18, 19], [19, 20]], dtype=torch.long)


def clones(module, N):
    # copy.deepcopy硬拷贝函数
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # [batch, max_len, 1]
        std = x.std(-1, keepdim=True)  # [batch, max_len, 1]
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # sublayer是layer的一个子结构
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 将残差结构复制两份
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 第一层：self_attn
        return self.sublayer[1](x, self.feed_forward)  # 第二层：feed_forward


def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 极小值填充

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # WQ=WK=WV=W0:[512, 512]
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):  # q, k, v就是传进的x
        # q=k=v: [batch_size, max_len, d_model]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 第一步：将q,k,v分别与Wq，Wk，Wv矩阵相乘, Wq=Wk=Wv: [512,512]
        # 第二步：将获得的Q、K、V进行维度切分, [batch_size,max_length,8,64]
        # 第三部：交换纬度, [batch_size,8,max_length,64]
        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]
        # q*wq, k*wk, v*wv, self.linears调用了3层

        # 得到Q, K, V之后开始做scaled dot-product attention
        # Query=Key=Value: [batch_size, 8, max_len, 64]
        # x: 输出， self.attn: 计算得到的attention
        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)
        # 纬度交换还原： [batch_size, max_length, 512]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 与W0大矩阵相乘： [batch_size, max_len, 512]
        return self.linears[-1](x)  # 调用第四层linear


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # [512, 2048]
        self.w_1 = nn.Linear(d_model, d_ff)
        # [2048, 512]
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True]]])


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=21):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = GraphConv(in_features, in_features * 2)
        self.gconv2 = GraphConv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class GraFormer(nn.Module):
    def __init__(self, adj, hid_dim=128, coords_dim=(2, 3), num_layers=4, nodes_group=None,
                 n_head=4,  dropout=0.1, n_pts=21):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        self.adj = adj

        _gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        if nodes_group is None:
            for i in range(num_layers):
                _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                                hid_dim=hid_dim, p_dropout=0.1))
                _attention_layer.append(EncoderLayer(dim_model, c(attn), c(gcn), dropout))

        # self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3, K=2)

    def forward(self, x, mask):
        out = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            out = self.atten_layers[i](out, mask)
            out = self.gconv_layers[i](out)
        out = self.gconv_output(out, self.adj)
        return out


class mse2d_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, uv_pre, uv_gt):
        e2d = torch.mean(torch.pow((uv_gt - uv_pre), 2))
        return e2d


if __name__ == '__main__':
    adj = adj_mx_from_edges(num_pts=21, edges=gan_edges, sparse=False)
    model = GraFormer(adj=adj, hid_dim=128)
    x = torch.zeros((1, 21, 2))
    print(model(x, src_mask))

