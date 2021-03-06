#!/usr/local/bin/python
'''
Reference: Tong Zhao et al., "Leveraging Social Connections to Improve 
Personalized Ranking for Collaborative Filtering." in CIKM 2014
@author: wubin
'''
from util.DataIterator import DataIterator
from util.Tool import csr_to_user_dict
import os
import tensorflow as tf
import numpy as np
from time import time
from util import Learner, Tool, DataGenerator, create_logger
from model.AbstractRecommender import AbstractRecommender
from util import timer
import scipy.sparse as sp

class LightGCN_kernel(AbstractRecommender):
    def __init__(self,sess,dataset, conf):
        super(LightGCN_kernel, self).__init__(dataset, conf)
        self.logger = create_logger(conf)
        self.learning_rate = float(conf["learning_rate"])
        self.embedding_size = int(conf["embedding_size"])
        self.learner = conf["learner"]
        self.num_epochs= int(conf["epochs"])
        self.batch_size = int(conf["batch_size"])
        self.verbose = conf["verbose"]
        self.reg = float(conf["reg"])
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.weight_size = conf["weight_size"]
        self.n_layers = len(self.weight_size)
        self.data_name = conf["data.input.dataset"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.R = dataset.train_matrix

        self.graph = dataset.train_matrix.tolil()
        self.norm_adj = self.get_adj_mat()
        self.sess = sess
        self.w  = conf['w']
        self.logger.info(conf)
    def get_adj_mat(self):
            
        try:
            pre_adj_mat=sp.load_npz('dataset/%s/pre_adj_mat.npz'%self.data_name)
        except:
            adj_mat = self.create_adj_mat()

            #adj_mat= adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()

        return pre_adj_mat
    
    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr()
    
    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
#             self.anchor_in = tf.placeholder(tf.float32,shape=[self.n_layers+1,self.n_layers+1],name='anchor')

    def _convert_sp_mat_to_sp_tensor(self, X):
        #print('************************')
        #print(X)
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    
    def kernel(self, latent):
        latent=tf.nn.l2_normalize(latent)
        cos=tf.matmul(latent,latent,transpose_b=True)
        cos=tf.clip_by_value(cos,-1,1)
        d=tf.acos(cos)
        d_one=tf.reshape(d,[-1])
        threshold=tf.nn.top_k(d_one,int(self.embedding_size**2*self.w)).values[-1]
        g=tf.where(tf.greater(threshold,d),tf.ones_like(d),tf.zeros_like(d))
        anchor = tf.multiply(tf.subtract(tf.ones_like(d),tf.square(d)),g)
        anchor_sum=tf.expand_dims(tf.reduce_sum(anchor,-1),-1)
#         return tf.cond(tf.count_nonzero(anchor,dtype=tf.int32)>tf.constant(self.n_layers+1),
#                        lambda:anchor/anchor_sum,lambda:tf.ones_like(anchor))
        return anchor/anchor_sum
    
    def _create_lightgcn_embed(self):

        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['user_embedding'] = tf.Variable(initializer([self.num_users, self.embedding_size]), name='user_embedding')
        self.weights['item_embedding'] = tf.Variable(initializer([self.num_items, self.embedding_size]), name='item_embedding')
        self.weights['W_mlp'] = tf.Variable(
        initializer([self.embedding_size, self.embedding_size]), name='W_mlp')
        self.weights['b_mlp'] = tf.Variable(
        initializer([1, self.embedding_size]), name='b_mlp')

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]
        norm_adj=self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        
        for n in range(0, self.n_layers):
            
            if n==0:
                ego_embeddings_s = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
#                 u_e,i_e=tf.split(ego_embeddings_s,[self.num_users, self.num_items], 0)
#                 R=tf.matmul(u_e,i_e,transpose_b=True)
                ego_embeddings_s = tf.matmul(ego_embeddings_s, self.weights['W_mlp'])
                s_matrix=tf.nn.softmax(ego_embeddings_s)
                norm_adj=tf.sparse_tensor_dense_matmul(norm_adj,s_matrix)
                norm_adj=tf.matmul(s_matrix,norm_adj,transpose_a=True)
                ego_embeddings = tf.matmul(s_matrix,ego_embeddings,transpose_a=True)
                
            else:
                ego_embeddings = tf.matmul(self.kernel(ego_embeddings),ego_embeddings)

#                 ego_embeddings = tf.matmul(norm_adj,ego_embeddings)
#                 u_e,i_e=tf.split(ego_embeddings,[self.num_users, self.num_items], 0)

        ego_embeddings=tf.matmul(s_matrix,ego_embeddings)
            
#         all_embeddings += [ego_embeddings]
#         self.all_embeddings_collect=tf.stack(all_embeddings,0)
        
#         self.anchor = self.kernel(tf.reshape(self.all_embeddings_collect,shape=[self.n_layers+1,-1]))
#         all_embeddings=tf.matmul(self.anchor_in,tf.reshape(self.all_embeddings_collect,shape=[self.n_layers+1,-1]))
#         all_embeddings=tf.reduce_mean(self.all_embeddings_collect,axis=0,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(tf.reshape(ego_embeddings,shape=[-1,self.embedding_size]), [self.num_users, self.num_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _create_variables(self):
        self.user_embeddings, self.item_embeddings = self._create_lightgcn_embed()
        self.emb=tf.concat((self.user_embeddings,self.item_embeddings),0)
    def _create_inference(self):
        with tf.name_scope("inference"):
            self.u_g_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
            self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
            self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.item_input_neg)
            self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False,
                                           transpose_b=True)
    def _create_loss(self):
        with tf.name_scope("loss"):
            self.pos_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
            neg_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings), axis=1)
            mf_loss = tf.reduce_mean(tf.nn.softplus(-(self.pos_scores - neg_scores)))
            
            self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.user_input)
            self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.item_input)
            self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.item_input_neg)
            
            
            regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
            regularizer = regularizer / self.batch_size
            emb_loss = self.reg * regularizer
            self.loss = mf_loss + emb_loss
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = Learner.optimizer(self.learner, self.loss, self.learning_rate)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
    
    #---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
#         self.w=np.ones((self.n_layers+1,self.n_layers+1))
        for epoch in range(1,self.num_epochs+1):
            # Generate training instances
            user_input, item_input_pos, item_input_neg = DataGenerator._get_pairwise_all_data(self.dataset)
            data_iter = DataIterator(user_input, item_input_pos, item_input_neg,
                                     batch_size=self.batch_size, shuffle=True)
            
            total_loss = 0.0
            training_start_time = time()
            for bat_users, bat_items_pos, bat_items_neg in data_iter:
                feed_dict = {self.user_input: bat_users,
                             self.item_input: bat_items_pos,
                             self.item_input_neg: bat_items_neg
                            }
                loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss/len(user_input),
                                                             time()-training_start_time))
#             if epoch<200:
#                 self.w=anchor_pre
            if epoch % 20== 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
        
        # params = self.sess.run([self.user_embeddings, self.item_embeddings])
        # with open("pretrained/%s_epochs=%d_embedding=%d_MF.pkl" % (self.dataset.dataset_name, self.num_epochs,self.embedding_size), "wb") as fout:
        #         pickle.dump(params, fout)
    
    @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run([self.user_embeddings, self.item_embeddings])
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_userids):
        if candidate_items_userids is None:
            user_embed = self._cur_user_embeddings[user_ids]
            ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        else:
            ratings = []
            for userid, items_by_userid in zip(user_ids, candidate_items_userids):
                user_embed = self._cur_user_embeddings[userid]
                items_embed = self._cur_item_embeddings[items_by_userid]
                ratings.append(np.squeeze(np.matmul(user_embed, items_embed.T)))
            
        return ratings