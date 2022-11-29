import torch
import torch.nn as nn
import numpy as np 

class Model(nn.Module):
    def __init__(self, config) -> None:
        super(Model, self).__init__()
        # embedding
        # input(batch_size, seq_len, vocab_size) output(batch_size, seq_len, embed_dim)
        self.embedding = nn.Embedding(config.vocab_size+2, config.embed_dim, padding_idx=config.vocab_size+1)
        # lstm:(input_size, hidden_size, num_layers, layers_num, 双向, )
        # input(batch_size, seq_len, embed_dim) output(batch_size, seq_len, hidden_size)
        self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, 
                            config.num_layers, bidirectional=True, 
                            batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x        # 去掉最后一维
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])    # 取最后的隐状态
        return out