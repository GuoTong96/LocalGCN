import faiss
import numpy as np
import random
import scipy.sparse as sp
from time import time
data='pet'
adj_mat=sp.load_npz('../dataset/%s/adj_mat.npz'%data).tolil()
emb_item=np.load('%s.npy'%data)
q=emb_item.astype(np.float32)
d=64
nlist=64
sample=64
cpu_index = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
index = faiss.IndexIVFFlat(gpu_index, d, nlist, faiss.METRIC_L2)
item_node=np.array(range(q.shape[0]))
index.train(q)
print(index.is_trained)
index.add(q)

_, I =index.search(q, sample)
subgraph_node=[]
for cluster in range(nlist):
    subgraph_node=np.setdiff1d(item_node,np.array(list(map((lambda x :x[cluster]),I))))

    sub_graph_adj = adj_mat
    sub_graph_adj[subgraph_node] = 0
    sub_graph_adj=sub_graph_adj.tocsr()
    sub_graph_adj = sub_graph_adj.multiply(sub_graph_adj.transpose())
    rowsum = np.array(sub_graph_adj.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_norm_adj = d_mat_inv.dot(sub_graph_adj)
    random_norm_adj = random_norm_adj.dot(d_mat_inv)
    print('generate random norm adjacency matrix%d.'%(cluster))
    random_adj_mat = random_norm_adj.tocsr()

    sp.save_npz('../dataset/%s/100faiss/s_random_adj_mat%d.npz'%(data,cluster), random_adj_mat)