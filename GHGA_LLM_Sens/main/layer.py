import math

import paddle
from paddle import nn
import numpy as np

class GCN(nn.Layer):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        stdv = 1. / math.sqrt(self.out_features)

        self.weight = self.create_parameter(shape=[self.in_features, self.out_features], dtype='float32',
                        default_initializer=nn.initializer.Uniform(low=-stdv, high=stdv))
        self.add_parameter('weight', self.weight)

        if bias:
            self.bias = paddle.create_parameter(shape=[self.out_features], dtype='float32',
                        default_initializer=nn.initializer.Uniform(low=-stdv, high=stdv))
            self.add_parameter('bias', self.bias)
        else:
            self.add_parameter('bias', None)

    def forward(self, adj, inputs, identity=False):
        if identity:
            return paddle.matmul(adj, self.weight)
        return paddle.matmul(adj, paddle.matmul(inputs, self.weight))

    def inference(self, inputs):
        return paddle.matmul(inputs, self.weight)


class GAT(nn.Layer):
    def __init__(self, in_features, out_features, dim=256,bias=True):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        stdv = 1. / math.sqrt(self.out_features)

        self.weight = self.create_parameter(shape=[dim, self.out_features], dtype='float32',
                        default_initializer=nn.initializer.Uniform(low=-stdv, high=stdv))
        self.add_parameter('weight', self.weight)
        if bias:
            self.bias = paddle.create_parameter(shape=[self.out_features], dtype='float32',
                        default_initializer=nn.initializer.Uniform(low=-stdv, high=stdv))
            self.add_parameter('bias', self.bias)
        else:
            self.add_parameter('bias', None)
        self.attention=AttentionLayer(self.out_features)

    def forward(self, adj, inputs, identity=False,att=True):
        # print(inputs)
        h1=nn.functional.linear(inputs, self.weight)
        if att:
            return self.attention(h1,adj)
        # if identity:
        #     return paddle.matmul(adj, self.weight)
        # return paddle.matmul(adj, paddle.matmul(inputs, self.weight))

    def inference(self, inputs):
        return paddle.matmul(inputs, self.weight)


class AttentionLayer(nn.Layer):
    def __init__(self, dim_features):
        super(AttentionLayer, self).__init__()

        self.dim_features = dim_features
        self.dropout = nn.Dropout(p=0.8)
        self.a1 = paddle.create_parameter(shape=[self.dim_features, 1], dtype='float32')
        nn.initializer.XavierNormal(self.a1)
        # self.a2 = paddle.create_parameter(shape=[self.dim_features, 1], dtype='float32')
        # nn.initializer.XavierNormal(self.a2)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input1, adj):
        h = input1
        h=self.dropout(h)
        e1 = paddle.matmul(h, self.a1)
        e = self.leakyrelu(e1)

        zero_vec = -9e15 * paddle.ones_like(e)

        attention = paddle.where(adj > 0, e, zero_vec)
        
        attention = nn.functional.softmax(attention)
        del zero_vec
        # print(h)
        h_prime = paddle.matmul(attention, h)

        return h_prime

    
class GCN_Att(nn.Layer):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN_Att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        stdv = 1. / math.sqrt(self.out_features)

        self.weight = self.create_parameter(shape=[self.in_features, self.out_features], dtype='float32',
                        default_initializer=nn.initializer.Uniform(low=-stdv, high=stdv))
        
        self.add_parameter('weight1', self.weight)
        self.a1 = paddle.create_parameter(shape=[self.out_features, 1], dtype='float32')
        nn.initializer.XavierNormal(self.a1)
        # self.a2 = paddle.create_parameter(shape=[self.out_features, 1], dtype='float32')
        # nn.initializer.XavierNormal(self.a2)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, input1,input2):
        # print(self.weight)
        h1=nn.functional.linear(input1, self.weight)
        h2=nn.functional.linear(input2, self.weight)
        h1=self.dropout(h1)
        h2=self.dropout(h2)
        e1 = paddle.matmul(h1, self.a1)
        e2 = paddle.matmul(h2, self.a1)
        e = self.leakyrelu(e1+e2)
        # print(e)
        zero_vec = -9e15 * paddle.ones_like(e)
        adj=np.eye(4293, dtype=np.float32)
        adj=paddle.to_tensor(adj,dtype='float32')
        # print(adj)
        attention = paddle.where(adj>0, e, zero_vec)
        attention = nn.functional.softmax(attention)
        del zero_vec
        del adj
        # print(attention)
        # print(h2)
        h_prime = paddle.matmul(attention, h2)

        return h_prime


