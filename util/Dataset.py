#!/usr/local/bin/python
import numpy as np
from util.Tool import randint_choice,get_data_format
import scipy.sparse as sp
import pandas as pd
import random as rd

class Dataset(object):
    def __init__(self, conf):
        """
        Constructor
        """
        path = conf["data.input.path"]
        dataset_name = conf["data.input.dataset"]
        self.separator = conf["data.convert.separator"]
        threshold = conf["data.convert.binarize.threshold"]
        evaluate_neg = conf["rec.evaluate.neg"]
        splitter_ratio = conf["data.splitterratio"]
        data_format=conf['data.column.format']
        path = path + dataset_name
        self.dataset_name = dataset_name
        columns = get_data_format(data_format)
        data_splitter = GivenData(path, dataset_name,columns, self.separator, threshold)

        self.train_matrix, self.test_matrix,  self.userids, self.itemids,self.num_r,self.sparsity = data_splitter.load_data()
   
        self.num_users, self.num_items = self.train_matrix.shape
        self.negative_matrix = self.get_negatives(evaluate_neg)
             
    def get_negatives(self, evaluate_neg):
        if evaluate_neg > 0:
            user_list = []
            neg_item_list = []
            for u in np.arange(self.num_users):
                items_by_u = self.train_matrix[u].indices.tolist() + self.test_matrix[u].indices.tolist()
                neg_items = randint_choice(self.num_items, evaluate_neg, replace=False, exclusion=items_by_u).tolist()
                neg_item_list.extend(neg_items)
                user_list.extend(len(neg_items)*[u])
            negatives = sp.csr_matrix(([1] * len(user_list), (user_list, neg_item_list)),
                                      shape=(self.num_users, self.num_items))
        else:
            negatives = None
        return negatives
     

class GivenData(object):
    def __init__(self, path, dataset_name, columns, separator, threshold):
        self.path = path
        self.dataset_name = dataset_name
        self.separator = separator
        self.threshold = threshold
        self.columns = columns
    def load_data(self):

        train_data = pd.read_csv(self.path+"/%s.train"%self.dataset_name, sep=self.separator, header=None, names=self.columns)
        test_data = pd.read_csv(self.path + "/%s.test"%self.dataset_name, sep=self.separator, header=None, names=self.columns)

        all_data = pd.concat([train_data, test_data])

        unique_user = all_data["user"].unique()
        unique_item = all_data["item"].unique()
        user2id = pd.Series(data=rd.shuffle(list(range(len(unique_user)))), index=unique_user)
        item2id = pd.Series(data=rd.shuffle(list(range(len(unique_item)))), index=unique_item)

        num_users = len(unique_user)
        num_items = len(unique_item)
        userids = user2id.to_dict()
        itemids = item2id.to_dict()


        train_matrix = sp.csr_matrix(([1]*len(train_data["user"]), (train_data["user"], train_data["item"])), shape=(num_users, num_items))
        test_matrix = sp.csr_matrix(([1]*len(test_data["user"]), (test_data["user"], test_data["item"])), shape=(num_users, num_items))  

        num_ratings = len(train_data["user"]) + len(test_data["user"])
        sparsity = 1 - num_ratings/(num_users*num_items)
        
        return train_matrix, test_matrix,  userids, itemids,num_ratings, sparsity
