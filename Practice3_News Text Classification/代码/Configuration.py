import os
import torch
import numpy as np



# 用于写超参数、文件名等
class Config():
    """
    说明：
        直接在Config里面写超参数、文件名
    """
    def __init__(self) -> None:
        self.train_set_path = "./训练集/train_set.csv/train_set.csv"
        self.test_set_path = "./测试集/test.csv/test_a.csv"
        save_dir = os.path.join('.', 'save')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir, 'save_model.pth')
        output_dir = os.path.join('.', '预测结果')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_csv_path = os.path.join(output_dir, 'submit.csv')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.vocab_size = 0 # 在utils中产生
        self.pad_size = 128  # log_2(vocab_size = 7000)  TODO:
        self.batch_size = 128   # TODO:
        self.num_epochs = 30    # TODO:
        self.lr = 1e-3
        self.dropout = 0.5      # 随机失活
        self.embed_dim = 300    # 词向量的维度
        self.hidden_size = 128  # lstm隐藏状态的维度
        self.num_layers = 2     # lstm共2层
        self.num_classes = 14