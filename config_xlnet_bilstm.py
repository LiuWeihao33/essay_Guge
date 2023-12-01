import numpy as np
import json
import tensorflow as tf
class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'test33_rebuttle'
        self.save_path = './saved_dict/' + self.model_name         # 模型训练结果
        self.log_path = './log/' + self.model_name
        self.dropout = 0.0                                            # 随机失活，可以自己尝试0,1，0.2等根据效果来选择
        self.require_improvement = 1000                                # 若超过1000batch效果还没提升，则提前结束训练,可以自己根据数据量大小，自己定义
        self.num_classes = 3                                            # 类别数，自定义
        self.class_list = ['Bad','Medium','Great']                             # 类别,自定义
        self.num_epochs =60                                           # epoch数,可以自己定义
        self.batch_size = 16                                          # mini-batch大小
        device = tf.device('cuda:1' if tf.test.is_gpu_available() else 'cpu')   # 设备

        self.learning_rate = 0.0001                                 # 学习率,可以根据模型的效果，自己定义学习率,1e-4,1e-5
        self.hidden_size = 256                                          # lstm隐藏层
        self.hidden_size_2 = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.embd = 768                                     #词向量维度
