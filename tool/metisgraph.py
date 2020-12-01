import pymetis
import scipy.sparse as sp
import numpy as np
import random
import networkx as nx
data='kdd'
adj = sp.load_npz('../dataset/%s/adj_mat.npz'%data)

l=[]
ll=[]
random.seed(2020)
adjj=adj.toarray()

for i in range(adjj.shape[0]):
    ll=np.unique(np.multiply(np.clip(adjj[i],0,1),range(adj.shape[1])))
    if adjj[i][0]==0:
        ll=list(ll[1:])
    else:
        ll=list(ll)
    l.append(ll)
(st, parts) = pymetis.part_graph(adjacency=l,nparts=10)
clusters = list(set(parts))
cluster_membership = {node:membership for node, membership in enumerate(parts)}
for cluster in [43,58,92,72,98]:
    sub_graph_adj = adj.tolil()
    subgraph_clu = random.sample(clusters,random.randint(1,10))
    subgraph_node=[]
    for n in range(adj.shape[0]):
        for clu in subgraph_clu:
            if cluster_membership[n]==clu:
                subgraph_node.append(n)
    subgraph_node=list(set(subgraph_node))
    sub_graph_adj[subgraph_node] = 0
    sub_graph_adj=sub_graph_adj.tocsr()
    sub_graph_adj = sub_graph_adj.multiply(sub_graph_adj.transpose())
    rowsum = np.array(sub_graph_adj.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    random_norm_adj = d_mat_inv.dot(sub_graph_adj)
    random_norm_adj = random_norm_adj.dot(d_mat_inv)
    print('generate random norm adjacency matrix%d.'%cluster)
    random_adj_mat = random_norm_adj.tocsr()

    sp.save_npz('../dataset/%s/100metis/s_random_adj_mat%d.npz'%(data,cluster), random_adj_mat)
    
