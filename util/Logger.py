#!/usr/local/bin/python
"""
@author: Zhongchuan Sun
"""
from configparser import ConfigParser
from collections import OrderedDict
import logging
import time
import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.logger = logging.getLogger("/home/guotong03/.jupyter/neurec/NeuRec")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # show on screen
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)

        # add two Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()


def create_logger(conf):

    config = ConfigParser()
    config.read("/home/guotong03/.jupyter/neurec/NeuRec.properties")
    lib_config = OrderedDict(config._sections["default"].items())
    model_name = conf["recommender"]

    model_config_path = os.path.join("/home/guotong03/.jupyter/neurec/conf", model_name + ".properties")
    config.read(model_config_path)
    model_config = OrderedDict(config._sections["hyperparameters"].items())

    data_name = lib_config["data.input.dataset"]

    log_dir = os.path.join("/home/guotong03/.jupyter/neurec/log", conf["data.input.dataset"], conf['recommender'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if model_name not in ['FastLightGCN','Local','Local_end','LightGCN_kernel']:
        logger_name = "%s_%s_reg%f_lr%f.log" % (conf["data.input.dataset"], conf['recommender'], conf['reg'], conf['learning_rate'])
    elif model_name in ['LightGCN_kernel']:
        logger_name = "%s_%s_reg%f_lr%f_w%f.log" % (conf["data.input.dataset"], conf['recommender'], conf['reg'], conf['learning_rate'],conf['w'])
    else:
        logger_name = "%s_%s_reg%f_lr%f_alpha%d_d%f_adj:%s.log" % (conf["data.input.dataset"], conf['recommender'], conf['fast_reg'], conf['learning_rate'],conf['r_alpha'],conf['d'],conf['adj_type'])
        
    logger_name = os.path.join(log_dir, logger_name)
    logger = Logger(logger_name)


    return logger


