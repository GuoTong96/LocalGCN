#!/usr/local/bin/python
import os
import random
import numpy as np
import tensorflow as tf
import importlib
from util import Configurer, Tool, Dataset

np.random.seed(2018)
random.seed(2018)
tf.set_random_seed(2017)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    conf = Configurer()
    recommender = str(conf["recommender"])

    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    dataset = Dataset(conf)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        
        my_module = importlib.import_module("model." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
