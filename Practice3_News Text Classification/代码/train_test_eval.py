import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn import metrics 
import pandas as pd 
import time 
from utils import get_time_dif
import numpy as np 

def init_net_params(model):
    for name, param in model.named_parameters():
        if 'embedding' not in name:
            if 'weight' in name:
                # 正态分布赋初值
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                pass



def train(config, model, train_iter):
    # 设置时间
    start_time = time.time()
    # 开启training模式
    model.train()
    # optimizer
    optimizer =  torch.optim.Adam(model.parameters(), lr=config.lr)


    for epoch in range(config.num_epochs):
        # 训练每一个epoch中当前batch次数
        batch_count = 0
        # added_loss = 0
        print('Epoch {}/{}:'.format(epoch+1, config.num_epochs))
        for batch, y in train_iter:
            y_pred = model(batch)
            model.zero_grad()
            loss = F.cross_entropy(y_pred, y)   # 交叉熵损失函数
            loss.backward()
            optimizer.step()                    # 更新参数
            if batch_count % 100 == 0:
                pred_labels = torch.argmax(y_pred.data, axis=1).cpu()    # 放到cpu上训练
                print(pred_labels.data)
                true_labels = y.data.cpu()
                # 计算准确度
                train_acc = metrics.accuracy_score(true_labels, pred_labels)
                # 保存模型
                torch.save(model.state_dict(), config.save_path)
                # 时间记录
                time_dif = get_time_dif(start_time)

                msg = 'Iter: {0:>5}, Train Loss: {1:>5.2}, Train_acc: {2:>6.2%}, Time: {3}'
                print(msg.format(batch_count, loss.item(), train_acc, time_dif))
                model.train()
            batch_count += 1
    torch.save(model.state_dict(), config.save_path)
    return model

def test(config, model, test_iter):
    # 提取模型
    model.load_state_dict(torch.load(config.save_path))
    # 模型调整为evaluate状态
    model.eval()
    # 计时开始
    start_time = time.time()
    
    # 下面要计算的是test_loss
    added_loss = 0
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for batch, y in test_iter:
            y_pred = model(batch)
            loss = F.cross_entropy(y_pred, y)
            added_loss += loss
            labels = y.data.cpu().numpy().tolist()
            preds = torch.argmax(y_pred, axis=1).cpu().numpy().tolist()
            pred_labels.extend(preds)
            true_labels.extend(labels)
    test_acc = metrics.accuracy_score(true_labels, pred_labels)
    test_loss = added_loss / len(test_iter)
    test_F1 = metrics.f1_score(true_labels, pred_labels, average='macro')
    time_dif = get_time_dif(start_time)
    # 生成message
    print("Now Test Outcome: ")
    message = "\tTest Loss: {0:>5.2}, Test Accuracy: {1:>6.2}, F1 Score: {2:>6.2}, Time: {3}"
    message = message.format(test_loss, test_acc, test_F1, time_dif)
    print(message)

    # 产生Test预测结果
    def generate_submit_outcome(output_csv_path, pred_labels):
        label_dic = {'label': pred_labels}
        try:
            df = pd.DataFrame(label_dic)
            df.to_csv(output_csv_path)
        except Exception:
            pass
    # 输出结果
    generate_submit_outcome(config.output_csv_path, pred_labels)

            