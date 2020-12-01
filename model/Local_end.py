#!/usr/local/bin/python
from model.AbstractRecommender import AbstractRecommender
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import multiprocessing
from util import timer
from time import time
from util import l2_loss, inner_product, create_logger
import random


class Local_end(AbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(Local_end, self).__init__(dataset, config)
        self.logger = create_logger(config)
        self.logger.info(config)
        # argument settings
        self.model_type = config['recommender']
        self.epoch = config["epoch"]
        self.adj_type = config["adj_type"]
        self.alg_type = config["alg_type"]
        self.n_users, self.n_items = dataset.num_users, dataset.num_items
        self.R = dataset.train_matrix
        self.dataset = dataset
        self.data_name = config["data.input.dataset"]
        self.lr = config["learning_rate"]
        self.emb_dim = config["embed_size"]
        self.weight_size = config["weight_size"]
        self.node_dropout_flag = config["node_dropout_flag"]
        self.node_dropout = config["node_dropout"]
        self.mess_dropout = config["mess_dropout"]
        self.n_layers = len(self.weight_size)
        self.r_alpha = config["r_alpha"]
        self.fast_reg = config["fast_reg"]
        self.localmodel = config['localmodel']
        self.bw = config['d']
        self.sess = sess
        self.verbose=config['verbose']

        plain_adj, norm_adj, mean_adj, pre_adj, random_adj = self.get_adj_mat()
        if config["adj_type"] == 'plain':
            self.norm_adj = plain_adj
            print('use the plain adjacency matrix')
        elif config["adj_type"] == 'norm':
            self.norm_adj = norm_adj
            print('use the normalized adjacency matrix')
        elif config["adj_type"] == 'gcmc':
            self.norm_adj = mean_adj
            print('use the gcmc adjacency matrix')
        elif config["adj_type"] == 'pre':
            self.norm_adj = pre_adj
            print('use the pre adjcency matrix')
        else:
            self.pre_adj = pre_adj
            self.norm_adj = random_adj
            print('use the random adjacency matrix')

    def get_adj_mat(self):
        try:
            adj_mat = sp.load_npz('dataset/%s/adj_mat.npz'%self.data_name)
            norm_adj_mat = sp.load_npz('dataset/%s/norm_adj_mat.npz'%self.data_name)
            mean_adj_mat = sp.load_npz('dataset/%s/mean_adj_mat.npz'%self.data_name)
        except:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz('dataset/%s/adj_mat.npz'%self.data_name, adj_mat)
            sp.save_npz('dataset/%s/norm_adj_mat.npz'%self.data_name, norm_adj_mat)
            sp.save_npz('dataset/%s/mean_adj_mat.npz'%self.data_name, mean_adj_mat)
        try:
            pre_adj_mat = sp.load_npz('dataset/%s/pre_adj_mat.npz'%self.data_name)
        except:
            # adj_mat = adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz('dataset/%s/pre_adj_mat.npz'%self.data_name,pre_adj_mat)
        random_adj=[]
        for cluster in range(self.localmodel):
            random_adj.append(self._convert_sp_mat_to_sp_tensor(sp.load_npz('dataset/%s/100%s/s_random_adj_mat%d.npz'%(self.data_name,self.adj_type,cluster))))


        return adj_mat, norm_adj_mat, mean_adj_mat, pre_adj_mat ,random_adj

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def _init_weights(self):
        with tf.variable_scope('lightgcn'):
            all_weights = dict()

            initializer = tf.contrib.layers.xavier_initializer()
#             initializer = tf.random_normal_initializer(stddev=0.1)
            all_weights['user_embedding'] = tf.get_variable(name="user_embedding", shape=[self.n_users, self.emb_dim],
                                  initializer=initializer)
            all_weights['item_embedding'] = tf.get_variable(name="item_embedding", shape=[self.n_items, self.emb_dim],
                                  initializer=initializer)
            all_weights['local_user_embedding'] = tf.get_variable(name="local_user_embedding", shape=[self.n_users, self.emb_dim],
                                  initializer=initializer)
            all_weights['local_item_embedding'] = tf.get_variable(name="local_item_embedding", shape=[self.n_items, self.emb_dim],
                                  initializer=initializer)
            print('using xavier initialization')

        return all_weights


    def _create_lightgcn_embed(self,adj,ego_embeddings):


        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            ego_embeddings=tf.sparse_tensor_dense_matmul(adj, ego_embeddings)

            all_embeddings += [ego_embeddings]

        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return all_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        try:
            return tf.SparseTensor(indices, coo.data, coo.shape)
        except:
            print(indices)
            print(coo.data)
            print(coo.shape)

            
#     def kernel(self, latent, n):
#         cos=tf.matmul(latent,latent,transpose_b=True)
#         cos=tf.clip_by_value(cos,-1,1)
#         d=tf.acos(cos)
#         d_one=tf.reshape(d,[-1])
#         threshold=tf.nn.top_k(d_one,int(n*self.bw)).values[-1]
#         g=tf.where(tf.greater(threshold,d),tf.ones_like(d),tf.zeros_like(d))
#         anchor = tf.multiply(tf.subtract(tf.ones_like(d),tf.square(d)),g)
#         return tf.cond(tf.count_nonzero(anchor,dtype=tf.int32)>=tf.constant(self.localmodel),
#                        lambda:anchor,lambda:tf.ones_like(anchor))
    def kernel(self,latent):
        from sklearn.preprocessing import normalize
        n_latent = normalize(latent,axis=1)
        n_latent=latent
        cos=np.matmul(n_latent,n_latent.T)
        cos=np.clip(cos,-1,1)
        d=np.arccos(cos)
        m=np.zeros(d.shape)
        d_one=d.reshape(-1)
        
        threshold=np.sort(d_one)[int(-self.localmodel**2*self.bw)]
        m[d<threshold]=1
        anchor=np.multiply(np.subtract(np.ones(d.shape),np.square(d)),m)
        anchor_sum=anchor.sum(1).reshape(anchor.shape[0],-1)
        a=anchor/anchor_sum
        a[np.isnan(a)]=0
        return a
#         return anchor
        
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)
    
    def _init_constant(self):
        # interaction information
        user_item_idx = [[u, i] for (u, i), r in self.dataset.train_matrix.todok().items()]
        user_idx, item_idx = list(zip(*user_item_idx))

        self.user_idx = tf.constant(user_idx, dtype=tf.int32, shape=None, name="user_idx")
        self.item_idx = tf.constant(item_idx, dtype=tf.int32, shape=None, name="item_idx")

    def build_graph(self):
        '''
                *********************************************************
                Create Placeholder for Input Data & Dropout.
                '''
        # placeholder definition
        self.users_ph = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items_ph = tf.placeholder(tf.int32, shape=(None,))
        self.anchor = tf.placeholder(tf.float32,shape=(self.localmodel,self.localmodel))
        
        for i in range(self.localmodel):
            exec("self.model_adj%d = tf.sparse_placeholder(tf.float32,name='model_adj%d')"%(i,i))

        self.node_dropout_ph = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout_ph = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()
        self._init_constant()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        for n_cluster in range(self.localmodel):
            self.all_embeddings_temp = self._create_lightgcn_embed(self.norm_adj[n_cluster],ego_embeddings)
            if n_cluster==0:
                self.all_embeddings=[self.all_embeddings_temp]
                continue
            self.all_embeddings += [self.all_embeddings_temp]
        self.all_embeddings = tf.stack(self.all_embeddings,1)
        self.latent =tf.reduce_mean(self.all_embeddings,axis=0,keepdims=False)

        self.all_embeddings = tf.reduce_mean(self.all_embeddings,axis=1,keepdims=False)
#         self.all_embeddings = tf.reduce_max(self.all_embeddings, reduction_indices=[1])
        self.ua_embeddings, self.ia_embeddings = tf.split(tf.reshape(self.all_embeddings,(self.n_users+self.n_items,self.emb_dim)), [self.n_users, self.n_items], 0)
        
        
        local_embeddings = tf.concat([self.weights['local_user_embedding'], self.weights['local_item_embedding']], axis=0)
        for n_cluster in range(self.localmodel):
            self.local_embeddings_temp = self._create_lightgcn_embed(self.norm_adj[n_cluster],local_embeddings)
            if n_cluster==0:
                self.local_embeddings_all=[self.local_embeddings_temp]
                continue
            self.local_embeddings_all += [self.local_embeddings_temp]
        self.local_embeddings_all = tf.stack(self.local_embeddings_all,0)
        self.local_embeddings_all = tf.matmul(self.anchor,tf.reshape(self.local_embeddings_all,[self.localmodel,-1]))
        self.local_embeddings_all = tf.reduce_mean(self.local_embeddings_all,axis=0,keepdims=False)
        self.local_u,self.local_i=tf.split(tf.reshape(self.local_embeddings_all,(self.n_users+self.n_items,self.emb_dim)),[self.n_users, self.n_items], 0)
        

        """
        *********************************************************
        Inference for the testing phase.
        """
        # for prediction
        self.item_embeddings_final = tf.Variable(tf.zeros([self.n_items, self.emb_dim]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.n_users, self.emb_dim]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.ua_embeddings),
                           tf.assign(self.item_embeddings_final, self.ia_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.users_ph)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False,
                                       transpose_b=True)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.user_idx)
        self.i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.item_idx)

        """
        *********************************************************
        Generate Predictions & Optimize via fast loss.
        """
        term1 = tf.matmul(self.ua_embeddings, self.ua_embeddings, transpose_a=True)
        term2 = tf.matmul(self.ia_embeddings, self.ia_embeddings, transpose_a=True)
        loss1 = tf.reduce_sum(tf.multiply(term1, term2))

        user_embed = tf.nn.embedding_lookup(self.ua_embeddings, self.user_idx)
        item_embed = tf.nn.embedding_lookup(self.ia_embeddings, self.item_idx)
        pos_ratings = inner_product(user_embed, item_embed)

        loss1 += tf.reduce_sum((self.r_alpha - 1) * tf.square(pos_ratings) - 2.0 * self.r_alpha * pos_ratings)
        # reg
        reg_loss = l2_loss(self.u_g_embeddings_pre, self.i_g_embeddings_pre)

        self.loss = loss1 + self.fast_reg * reg_loss

        self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        self.local_u_pre = tf.nn.embedding_lookup(self.weights['local_user_embedding'], self.user_idx)
        self.local_i_pre = tf.nn.embedding_lookup(self.weights['local_item_embedding'], self.item_idx)
        local_term1 = tf.matmul(self.local_u, self.local_u, transpose_a=True)
        local_term2 = tf.matmul(self.local_i, self.local_i, transpose_a=True)
        loss2 = tf.reduce_sum(tf.multiply(local_term1, local_term2))

        local_user_embed = tf.nn.embedding_lookup(self.local_u, self.user_idx)
        local_item_embed = tf.nn.embedding_lookup(self.local_i, self.item_idx)
        local_pos_ratings = inner_product(local_user_embed, local_item_embed)

        loss2 += tf.reduce_sum((self.r_alpha - 1) * tf.square(local_pos_ratings) - 2.0 * self.r_alpha * local_pos_ratings)
        # reg
        reg_loss = l2_loss(self.local_u_pre, self.local_i_pre)

        self.local_loss = loss2 + self.fast_reg * reg_loss

        self.local_opt = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.local_loss)
    
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        n,best=0,0.0
        _precision,_recall,_map,_ndcg=[],[],[],[]
        self.w=np.ones((self.localmodel,self.localmodel))

#         for epoch in range(400):
#             _, loss, latent = self.sess.run([self.opt, self.loss,self.latent],feed_dict={self.anchor:self.w})
#         self.w=self.kernel(latent)
#         print('latent done, trian local model')
#         np.save('latent',self.w)
        for epoch in range(self.epoch):
            start = time()
            _, loss, latent = self.sess.run([self.opt, self.loss, self.latent],feed_dict={self.anchor:self.w})

#             _, loss = self.sess.run([self.local_opt, self.local_loss],feed_dict={self.anchor:self.w})
            end = time()

            print('anchor_epoch%d: loss = %f time = %fs' % (epoch, loss,end - start))
            if epoch % self.verbose == 0 :

                result = self.evaluate_model()
                self.logger.info("epoch %d:\t%s\tbest pre:%f" % (epoch, result,best))
                result=result.split('\t')
                _precision.append(float(result[0]))
                _recall.append(float(result[1]))
                _map.append(float(result[2]))
                _ndcg.append(float(result[3]))
                if best<float(result[0]):
                    best=float(result[0])
                    n=0
                else:
                    n+=1
                    if n>=500/self.verbose:
                        self.logger.info("bestresult:\t%f\t%f\t%f\t%f" % (max(_precision),max(_recall),max(_map),max(_ndcg)))
                        break
                
    @timer
    def evaluate_model(self):
        self.sess.run(self.assign_opt,feed_dict={self.anchor:self.w})
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, items=None):
        feed_dict = {self.users_ph: user_ids,self.anchor:self.w,
                     self.node_dropout_ph: [0.] * len(self.weight_size),
                     self.mess_dropout_ph: [0.] * len(self.weight_size)}
        i_rate_batch = self.sess.run(self.batch_ratings, feed_dict=feed_dict)

        return i_rate_batch
