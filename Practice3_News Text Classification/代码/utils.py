import os
import pandas as pd
import torch
import numpy as np
import pickle as pkl 
from tqdm import tqdm 
import time
from datetime import timedelta
from collections import Counter

# 通过已经得到的corpus生成vocab列表
def read_csv(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df

def get_labels(dataframe):
    labels = np.array(dataframe['label'])
    return labels

def get_corpus(dataframe):
    texts = list(dataframe['text'])
    corpus = []
    # 最终结果corpus=每行为一个文本的内容，每一个文本内容为数字列表，因此为二维
    # 注意，corpus是不等长的
    for i in range(len(dataframe)):
        line_lst = texts[i].split()
        corpus.append([int(item) for item in line_lst])
    
    return corpus

def get_vocab_size(corpus):
    """
    args:
        corpus: 2-d list
    return:
        vocab_size
    """
    vocab_size = max([max(row) for row in corpus])
    return vocab_size

def build_dataset(config):
    """
    args:
        config: arguments for configurations like file_path
    do-something:
        1.get labels (list for each text)
        2.get corpus (list for each text)
        3.get vocab_len
        4.pad and cut to fit the size of pad_size pre-defined
    returns:
        1.train data_set shape like:[([pad_size], label(int)), ([], *), ...]
        2.test data_set
        3.vocab_len
    """
    train_df = read_csv(config.train_set_path)
    test_df = read_csv(config.test_set_path)
    train_label_lst = get_labels(train_df)
    # test_label_lst = get_labels(test_df)
    test_label_lst = np.zeros((len(test_df),),dtype=int)
    train_corpus_lst = get_corpus(train_df)
    test_corpus_lst = get_corpus(test_df)
    config.vocab_size = get_vocab_size(train_corpus_lst)

    def load_dataset(corpus_lst, label_lst, vocab_size, pad_size):
        contents = []
        for i in range(len(corpus_lst)):
            content = corpus_lst[i]
            seq_len = len(content)
            if seq_len < pad_size:
                content.extend([vocab_size+1]*(pad_size-seq_len))
            else:
                content = content[:pad_size]
                seq_len = pad_size
            label = label_lst[i]
            contents.append((content, int(label), seq_len))
            # contents.append((content, int(label)))
        
        return contents
    train_set = load_dataset(train_corpus_lst, train_label_lst, config.vocab_size, config.pad_size)
    test_set = load_dataset(test_corpus_lst, test_label_lst, config.vocab_size, config.pad_size)
    config.vocab_size += 1
    return train_set, test_set

class DatasetIterator():
    """
    __init__
        本质上是在进行切分batches
    """
    def __init__(self, batches, batch_size, device) -> None:
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False        # 如果不整除 有残留的话，就是True
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 当前已经是取到第几批了
        self.device = device
        
    def _to_tensor(self, datas):
        x = torch.LongTensor([data[0] for data in datas]).to(self.device)
        y = torch.LongTensor([data[1] for data in datas]).to(self.device)
        seq_len = torch.LongTensor([data[2] for data in datas]).to(self.device)
        return (x, seq_len), y
    
    def __next__(self):

        if self.residue and self.index == self.n_batches:   # 最后一组且可能残留
            batches = self.batches[self.index * self.batch_size : len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:    # 已经取走了所有的batch
            self.index = 0
            raise StopIteration
        
        else:   # 取走前面普通的每组
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    time_dif = time.time() - start_time
    return timedelta(seconds=int(round(time_dif)))