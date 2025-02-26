import math
import os
import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from main.layer import GCN, GAT,GCN_Att

paddle.set_default_dtype('float32')
import time



class ETGAT(nn.Layer):
    def __init__(self, in_features_dim, out_features_dim, train_idx, params):
        super(ETGAT, self).__init__()
        self.threshold = params.threshold
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.drop_out = params.drop_out
        self.train_idx = train_idx

        self.GCNw=GAT(self.out_features_dim[0], self.out_features_dim[0])
        # self.GCNw = GCN(self.out_features_dim[0], self.out_features_dim[0])
        self.GATw = GAT(self.in_features_dim[0], self.out_features_dim[0],200)
        self.GCNc = GAT(self.out_features_dim[0], self.out_features_dim[0])
        # self.GCNc = GCN(self.out_features_dim[0], self.out_features_dim[0])
        self.GATc = GAT(self.in_features_dim[-1], self.out_features_dim[0],200)
        # self.GCN2 = GCN(self.out_features_dim[0], self.out_features_dim[0])
        # self.GAT = GCN_Att(456, self.out_features_dim[1])
        self.complinears = nn.Linear(self.in_features_dim[1], self.out_features_dim[1])
        # self.final_GCN = GCN(self.in_features_dim[2], self.out_features_dim[2])
        self.final_GCN = GAT(self.in_features_dim[2], self.out_features_dim[2])
        self.final_GAT = GAT(self.out_features_dim[2], self.out_features_dim[2])
        # self.final_GCN2 = GCN(self.out_features_dim[2], self.out_features_dim[2])
        self.FC = nn.Linear(in_features_dim[3], self.out_features_dim[3])

    def forward(self, adj, feature):
        t = time.time()
        refined_text_input=[]
        character_embedding=self.GCNc(adj['char'], F.relu(self.GATc(adj['char'], feature['char_emb'], identity=True)))
        character_embedding=paddle.concat([
            F.dropout(character_embedding, p=self.drop_out, training=self.training), feature['char_emb']], axis=-1)
        ce=paddle.matmul(adj['q2c'], character_embedding, ) / (
                    paddle.sum(adj['q2c'], axis=1, keepdim=True) + 1e-9)
        refined_text_input.append(ce/(paddle.linalg.norm(ce,p=2, axis=-1, keepdim=True)+1e-9))
        word_embedding = self.GCNw(adj['word'], F.relu(self.GATw(adj['word'], feature['word_emb'], identity=True)))
        word_embedding = paddle.concat([
            F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], axis=-1)
        we=paddle.matmul(adj['q2w'], word_embedding, ) / (
                    paddle.sum(adj['q2w'], axis=1, keepdim=True) + 1e-9)
        refined_text_input.append(we/(paddle.linalg.norm(we,p=2, axis=-1, keepdim=True)+1e-9))
        # print(refined_text_input)
        # refined_input=refined_text_input[0]
        refined_input=paddle.concat(refined_text_input,axis=1)
        # refined_input=self.GAT(refined_text_input[0],refined_text_input[1])
        Doc_features = refined_input[self.train_idx]
        # DocFea4ADJ = Doc_features / (paddle.linalg.norm(Doc_features, p=2, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_features = F.dropout(self.complinears(Doc_features), p=self.drop_out, training=self.training)
        cos_simi_total = paddle.matmul(Doc_features, Doc_features, transpose_y=True)
        refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float32'))
        refined_Doc_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        final_text_output = self.final_GCN(refined_Doc_adj, self.final_GAT(refined_Doc_adj, refined_Doc_features))
        final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        
        
        # word_embedding = self.GCN2(adj['word'], F.relu(self.GCN(adj['word'], feature['word'], identity=True)))
        # word_embedding = paddle.concat([
        #     F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], axis=-1)
        # refined_text_input = paddle.matmul(adj['q2w'], word_embedding, ) / (
        #             paddle.sum(adj['q2w'], axis=1, keepdim=True) + 1e-9)
        # Doc_features = self.complinears(refined_text_input[self.train_idx])
        # DocFea4ADJ = Doc_features / (paddle.linalg.norm(Doc_features, p=2, axis=-1, keepdim=True) + 1e-9)
        # refined_Doc_features = F.dropout(Doc_features, p=self.drop_out, training=self.training)
        # cos_simi_total = paddle.matmul(DocFea4ADJ, DocFea4ADJ, transpose_y=True)
        # refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float32'))
        # refined_Doc_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        # final_text_output = self.final_GAT(refined_Doc_adj, self.final_GCN(refined_Doc_adj, refined_Doc_features))
        # final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)
        # scores = self.FC(final_text_output)
        return scores
    
    def inference(self, adj, feature):
        t = time.time()
        word_embedding = self.GCN2(adj['word'], F.relu(self.GCN(adj['word'], feature['word'], identity=True)))
        word_embedding = paddle.concat([
            F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], axis=-1)
        refined_text_input = paddle.matmul(adj['q2w'], word_embedding) / (
                    paddle.sum(adj['q2w'], axis=1, keepdim=True) + 1e-9)
        Doc_features = self.complinears(refined_text_input)
        DocFea4ADJ = Doc_features / (paddle.linalg.norm(Doc_features, p=2, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_features = F.dropout(Doc_features, p=self.drop_out, training=self.training)
        cos_simi_total = paddle.matmul(DocFea4ADJ, DocFea4ADJ[self.train_idx], transpose_y=True)
        refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float32'))
        len_train = len(self.train_idx)
        supp_adj = paddle.sum((DocFea4ADJ * DocFea4ADJ), axis=-1, keepdim=True)
        supp_adj[self.train_idx] = 0
        refined_adj_tmp = paddle.concat([refined_adj_tmp, supp_adj], axis=-1)
        refined_Doc_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_adj, alpha_list = refined_Doc_adj.split([len_train, 1], axis=-1)
        Doc_train_adj = refined_Doc_adj[self.train_idx]
        Emb_train = self.final_GCN2(Doc_train_adj, self.final_GCN(Doc_train_adj, refined_Doc_features[self.train_idx]))
        Doc_output = paddle.matmul(refined_Doc_adj, Emb_train)
        emb_Doc_Feat = self.final_GCN2.inference(self.final_GCN.inference(refined_Doc_features))
        final_text_output = Doc_output + alpha_list * emb_Doc_Feat
        final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores
    
    def predict(self, adj, feature):
        t = time.time()
        
        refined_text_input=[]
        character_embedding=self.GCNc(adj['char'], F.relu(self.GATc(adj['char'], feature['char_emb'], identity=True)))
        character_embedding=paddle.concat([
            F.dropout(character_embedding, p=self.drop_out, training=self.training), feature['char_emb']], axis=-1)
        ce=paddle.matmul(adj['q2c'], character_embedding, ) / (
                    paddle.sum(adj['q2c'], axis=1, keepdim=True) + 1e-9)
        refined_text_input.append(ce/(paddle.linalg.norm(ce,p=2, axis=-1, keepdim=True)+1e-9))
        word_embedding = self.GCNw(adj['word'], F.relu(self.GATw(adj['word'], feature['word_emb'], identity=True)))
        word_embedding = paddle.concat([
            F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], axis=-1)
        we=paddle.matmul(adj['q2w'], word_embedding, ) / (
                    paddle.sum(adj['q2w'], axis=1, keepdim=True) + 1e-9)
        refined_text_input.append(we/(paddle.linalg.norm(we,p=2, axis=-1, keepdim=True)+1e-9))
        # print(refined_text_input)
        # refined_input=refined_text_input[0]
        refined_input=paddle.concat(refined_text_input,axis=1)
        # print(refined_input)
        # refined_input=self.GAT(refined_text_input[0],refined_text_input[1])
        Doc_features = refined_input
        # print(Doc_features)
        # DocFea4ADJ = Doc_features / (paddle.linalg.norm(Doc_features, p=2, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_features = F.dropout(self.complinears(Doc_features), p=self.drop_out, training=self.training)
        # print(refined_Doc_features)
        cos_simi_total = paddle.matmul(Doc_features, Doc_features, transpose_y=True)
        refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float32'))
        refined_Doc_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        # print(refined_Doc_adj)
        final_text_output = self.final_GCN(refined_Doc_adj, self.final_GAT(refined_Doc_adj, refined_Doc_features))
        # print(final_text_output)
        final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)
        # print(final_text_output)
        scores = self.FC(final_text_output)

        
        
        # word_embedding = self.GAT(adj['word'], F.relu(self.GCN(adj['word'], feature['word'], identity=True)))
        # word_embedding = paddle.concat([
        #     F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], axis=-1)
        # refined_text_input = paddle.matmul(adj['q2w'], word_embedding, ) / (
        #             paddle.sum(adj['q2w'], axis=1, keepdim=True) + 1e-9)
        # Doc_features = self.complinears(refined_text_input)
        # DocFea4ADJ = Doc_features / (paddle.linalg.norm(Doc_features, p=2, axis=-1, keepdim=True) + 1e-9)
        # refined_Doc_features = F.dropout(Doc_features, p=self.drop_out, training=self.training)
        # cos_simi_total = paddle.matmul(DocFea4ADJ, DocFea4ADJ, transpose_y=True)
        # refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float32'))
        # refined_Doc_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        # final_text_output = self.final_GAT(refined_Doc_adj, self.final_GCN(refined_Doc_adj, refined_Doc_features))
        # final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)
        # scores = self.FC(final_text_output)
        return scores