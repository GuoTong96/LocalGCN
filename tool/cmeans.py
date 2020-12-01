# from pylab import *
from numpy import *
import pandas as pd
import numpy as np
import operator
import math
import random
import scipy.sparse as sp

k = 100
MAX_ITER = 50
m = 2.00
dataset='cell'


def initializeMembershipMatrix(n):
    membership_mat=np.random.rand(n,k)
    membership_mat=membership_mat.T/np.sum(membership_mat.T,0)
    return membership_mat.T


def calculateClusterCenter(membership_mat,data,n):
    xraise=np.square(membership_mat)
    denominator=np.sum(xraise,axis=0)
    numerator=np.sum(np.multiply(xraise,data),axis=0)
    cluster_centers = np.true_divide(numerator,denominator)
    return cluster_centers


def updateMembershipValue(membership_mat, cluster_centers,data,n):

    dist,den=[],[]
    for j in range(k):
        
        dist +=[np.linalg.norm(data-cluster_centers[j],2,axis=1)]
    distance=np.vstack(dist)
    for j in range(k):
        den+=[np.square(distance/distance[j])]
    membership_mat=np.reciprocal(np.sum(den,0))
    return membership_mat.T


def getClusters(membership_mat,n):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansClustering(data):

    pre_adj = sp.load_npz('../dataset/%s/adj_mat.npz'%dataset)
    n=data.shape[0]
    membership_mat = initializeMembershipMatrix(n)
    curr = 0
    while curr <= MAX_ITER:  
        
        cluster_centers = calculateClusterCenter(membership_mat,data,n)
        membership_mat = updateMembershipValue(membership_mat, cluster_centers,data,n)
        
#         cluster_labels = getClusters(membership_mat,n)
        curr += 1
        if curr==10:
            print(membership_mat)
        
    for i in range(k):
        dd=membership_mat.T[i]
        adj=pre_adj.tolil()
        gate=np.random.choice(np.sort(dd)[int(0.1*dd.shape[0]):],1)
        adj[dd<gate]=0
        adj=adj.tocsr()
        adj=adj.multiply(adj.transpose())
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_norm_adj = d_mat_inv.dot(adj)
        random_norm_adj = random_norm_adj.dot(d_mat_inv)
        print('generate random norm adjacency matrix%d.'%i)
        random_adj_mat = random_norm_adj.tocsr()

        sp.save_npz('../dataset/%s/100cmeans/s_random_adj_mat%d.npz'%(dataset,i),random_adj_mat)
#         sp.save_npz('/home/guotong03/.jupyter/ppi/LightGCN-master/Data/off/100cmeans2/s_random_adj_mat%d.npz'%i,adj2)
#         sp.save_npz('/home/guotong03/.jupyter/ppi/neurec/dataset/%s/100cmeans3/s_random_adj_mat%d.npz'%(dataset,i),adj3)
    return membership_mat
embs=np.load('%s.npy'%dataset)
fuzzyCMeansClustering(embs)
