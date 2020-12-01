import scipy.sparse as sp
import numpy as np
import random as rd
data='pet'
adj_mat=sp.load_npz('../dataset/%s/adj_mat.npz'%data)
lenofnode = adj_mat.shape[0]
all_node = [node for node in range(lenofnode)]
# user_node=range(lenofnode)
# item_node=range(9534)
for cluster in range(0,100):
    sub_graph_adj = adj_mat.tolil()
    sub_len=rd.choice(range(lenofnode-int(len(all_node)/10)))
#     subgraph_lenofnode = rd.choice(range(1,len(all_node)))
   
    subgraph_node = rd.sample(all_node,sub_len)
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
    
    sp.save_npz('../dataset/%s/100random-90/s_random_adj_mat%d.npz'%(data,cluster), random_adj_mat)
