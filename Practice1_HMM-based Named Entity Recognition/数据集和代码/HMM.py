from itertools import count
import json
from re import L
import numpy as np
import os
import pandas as pd


"""
思路：
一、学习过程：
首先根据train数据,通过频数,获取得到各隐状态之间的概率转移矩阵A、定义每一个状态下得到各观测值的概率矩阵B、初始概率矩阵pi,这是一个统计的过程

二、推理过程：
编写维特比算法对输入的dev.json观测序列进行解码,得到预测状态序列

三、评估：
在步骤二中,得到的状态序列与dev.json的真实标签进行对比,计算出精确率P、召回率R和F1值

"""

# N：状态的有限集合
N = ['O', 'B-ADDRESS', 'I-ADDRESS', 'B-BOOK', 'I-BOOK', 'B-COM', 'I-COM', 'B-GAME', 'I-GAME', 'B-GOV', 'I-GOV', 'B-MOVIE', 'I-MOVIE', 'B-NAME', 'I-NAME', 'B-ORG', 'I-ORG', 'B-POS', 'I-POS', 'B-SCENE', 'I-SCENE']
# 状态的个数
n = 21
# 标签字典，key为标签类型， value为标签在N中的index
tagDict = {'O': 0, 'B-ADDRESS': 1, 'I-ADDRESS': 2, 'B-BOOK': 3, 'I-BOOK': 4, 'B-COM': 5, 'I-COM': 6, 'B-GAME': 7, 'I-GAME': 8, 'B-GOV': 9, 'I-GOV': 10, 'B-MOVIE': 11, 'I-MOVIE': 12, 'B-NAME': 13, 'I-NAME': 14, 'B-ORG': 15, 'I-ORG': 16, 'B-POS': 17, 'I-POS': 18, 'B-SCENE': 19, 'I-SCENE': 20}
labelDict = {'O': 0, 'address': 1, 'book': 2, 'company': 3, 'game': 4, 'government': 5, 'movie': 6, 'name': 7, 'organization': 8, 'position': 9, 'scene': 10}

# 下面定义隐概率转移矩阵 A (N*N)，表示i到j状态转移的概率
# 先将其初始化为0
A = {}
for i in range(n):
    A[i] = {}
    for j in range(n):
        A[i][j] = 0

# 下面定义每一个状态下得到观测值的矩阵 B(N*M) N为总共的状态数，M为总共的观测值种类数
B = {}
for i in range(n):
    B[i] = {}   #初始化每一个字典的内容

# 定义初始的状态分布 pi，并给定一个初值0
pi = {}
for i in range(n):
    pi[i] = 0

# 为了能够统计出pi，我们定义count_state，分别对每一个状态进行统计出现次数
count_state = {}
for i in range(n):
    count_state[i] = 0
# 统计各个状态下，出现各观测值的个数
count_s_o = {}
for i in range(n):
    count_s_o[i] = {}
# 统计从i状态到j状态的个数
count_sij = {}
for i in range(n):
    count_sij[i] = {}
    for j in range(n):
        count_sij[i][j] = 0
# 统计所有的状态
count_1_s = []
for i in range(n):
    count_1_s.append(0)


with open('./train.json', 'r', encoding='utf-8') as fr:
    for entry in fr:

        # print(entry) #其中entry 已经变为了str类型
        s = json.loads(entry)   # s当前是一个字典
        text_now = s['text']    # str,如一句话："生生不息CSOL生化狂潮让你填弹狂扫"
        label_now = s['label']  # label_now是label下的字典, 例如{"address": {"宝城": [[2, 3]]}, "name": {"梁京": [[5, 6]]}}

        # 创建当前的一句话中的每个字符（作为一个observation），其对应的隐含状态列表
        hidden_states = []
        for i in range(len(text_now)):
            hidden_states.append(0)    # 全部给定一个O类标注

        # 下面扫描所有的label
        # k: e.g. 'address'
        for k in label_now.keys():
            index1 = labelDict[k]   # e.g. address-> 1
            index2 = 2 * index1-1     # e.g. B-ADDRESS 的index就是adress的2倍-1, 2*1-1=1 
            # obs:观测字典，如{"汉堡": [[3, 4]], "汉诺威": [[16, 18]]}
            obs = label_now[k]

            #locations 如： [[3,4], [6,7], [10,11]]
            for locations in obs.values():
                list_location_ob_word = locations     # list_location_ob_word是定位的列表，可能有多次出现的位置，如[[3,4],[7,8]]
                for item in list_location_ob_word:
                    hidden_states[item[0]] = index2         # B-xxx

                    for i1 in range(item[0]+1, item[1]+1):
                        hidden_states[i1] = index2 + 1      # I-xxx
        
        # 统计第一个字符对应的隐藏状态的次数
        count_1_s[hidden_states[0]] += 1

        # 下面进行counter更新
        for obs_index in range(len(text_now)):
            # if hidden_states[obs_index] == 0:
            count_state[hidden_states[obs_index]] += 1      # 各状态出现次数计数
            # 如果之前没有统计过这个 状态-观测值的出现次数，现在就令其为 1
            if text_now[obs_index] not in count_s_o[hidden_states[obs_index]].keys():
                # 状态-观测值概率
                count_s_o[hidden_states[obs_index]][text_now[obs_index]] = 1
            else:   # 否则，出现次数需要加1
                count_s_o[hidden_states[obs_index]][text_now[obs_index]] += 1

        for obs_index in range(1, len(text_now)):
            # 状态i到状态j的转移频数
            count_sij[hidden_states[obs_index-1]][hidden_states[obs_index]] += 1


sum_1_s = sum(count_1_s)
for i in range(n):
    for j in range(n):
        # 状态转移概率矩阵估计（最大似然估计）
        A[i][j] = count_sij[i][j] / count_state[i]
    for ob in count_s_o[i].keys():
        # 状态-观测值 估计
        B[i][ob] = count_s_o[i][ob] / count_state[i]
    # 初始状态的概率分布 估计
    pi[i] = count_1_s[i] / sum_1_s



# 定义和初始化评估参数
TP = []
FP = []
# TN = []
FN = []

for i in range(21):
    TP.append(0)
    FP.append(0)
    # TN.append(0)
    FN.append(0)

# 下面对dev.json文件中的每一个数据text进行处理，通过观测各字符串用维特比算法，实现目的
with open('./dev.json', 'r', encoding='utf-8') as f_dev:
    # 下面按照每一个句子进行处理
    counter = 0
    for entry in f_dev:
        s = json.loads(entry)
        # seq 为一句待处理的数据
        seq = s['text']
        label_dict = s['label']

        length = len(seq)    # 待处理数据的总观测长度

        # 初始化derta
        derta = {}
        phi = {}
        for t in range(length):
            derta[t] = {}
            phi[t] = {}
            if t == 0:
                for i in range(n):
                    # 对于t时刻下的观测值，其是由i状态产生的概率是这样的
                    if seq[t] in B[i].keys():
                        derta[t][i] = pi[i] * B[i][seq[t]]
                    else:   # 如果之前从没有在i状态下观测到过B
                        # derta[t][i] = 0
                        # if i == 0:
                        derta[t][i] = pi[i] * (1 / count_state[i]) # 尝试用(1 / count_state[j])代替B[j][seq[t]]
                        # else:
                            # derta[t][i] = 0
                    # 这里实际上直接应用到了HMM的两个基本假设
                    # 1.马尔科夫的，则当前状态的转移概率只与前一步的状态有关
                    # 2.观测值是独立产生的，也就是只与当前的状态i有关，与其他均无关
                    phi[t][i] = 0

        # 更新迭代
        for t in range(1, length):
            for j in range(n):
                max_derta_1 = -100000
                max_phi = 0
                for i in range(n):  # 遍历上一个时刻，状态i
                    # Attention！ temp没有乘以B[j][O_t]
                    temp = derta[t-1][i] * A[i][j]
                    if max_derta_1 < temp:
                        max_derta_1 = temp
                        max_phi = i
                if seq[t] in B[j].keys():
                    derta[t][j] = max_derta_1 * B[j][seq[t]]
                else:   # 如果在j状态下从没观察到过当前的观测值
                    # derta[t][j] = 0
                    # if j == 0:    # TODO:当完全得到了模型评价之后，再来看如果没出现过当前的观测值应该怎么处理
                    derta[t][j] = max_derta_1 * (1 / count_state[j]) # 尝试用(1 / count_state[j])代替B[j][seq[t]]
                    # else:
                        # derta[t][j] = 0
                # 更新当前的phi值
                # derta[t][0] = 1
                phi[t][j] = max_phi
        
        # 结束遍历
        #首先定义最优序列q
        q = np.zeros(length)
        p_t_max = max(derta[length-1].values())
        for i in range(n):
            # 这里需要思考，如果出现了derta相同的情况，选择最大值如何选取，我是直接选取的从前到后出现的第一个最大的
            if derta[length-1][i] == p_t_max:
                q[length-1] = i
                break   # 这里我就直接让从前到后遍历到的第一个最大值为q
        
        # 回溯以得到路径
        for t in range(length-2 , -1, -1):
            q[t] = phi[t+1][q[t+1]]
            

        # #打印env中的预测隐藏状态序列
        # counter += 1
        # # if counter in range(10740, 10749):
        # print(counter,":[")
        # for item in q:
        #     print(list(tagDict.keys())[list(tagDict.values()).index(item)],end=' ')
        # print(']')


        hidden_states_dev = []  # 真实隐藏状态
        pred_states_dev = q     # 预测隐藏状态
        # pred_states_dev = []
        for i in range(length):
            hidden_states_dev.append(0) # 初始化真实隐藏状态
        # 下面生成该行text中每一个字符的真实label数据。
        # label_dict e.g. {"game":{"DotA":[[1, 4]]}}
        # label: e.g. "game"
        for label in label_dict.keys():
            index1 = labelDict[label]
            index2 = index1 * 2 - 1
            # list_locations e.g. [[1,2],[4,5]]
            for list_locations in label_dict[label].values():
                # location e.g. [1,2]
                for location in list_locations:
                    hidden_states_dev[location[0]] = index2
                    for loc_i in range(location[0]+1, location[1]+1):
                        hidden_states_dev[loc_i] = index2 + 1
        
        # 根据当前的句子情况，更新TP, FP, FN
        for i in range(length):
            hid_state = hidden_states_dev[i]
            pred_state = pred_states_dev[i]
            if hid_state == pred_state:
                TP[hid_state] += 1
            else:
                FP[int(pred_state)] += 1
                FN[hid_state] += 1
                # 不必计算TN


# 下面进行一些precision、recall和f1-score的计算
'''
思路：首先，遍历统计所有的预测标签和真实标签中 预测成功的有多少。
    分别用TP FP FN TN存储在每一个标签维度下的预测/真实|正确/错误的个数
    列表,index就是各类标签
'''

TP = np.array(TP)
FP = np.array(FP)
FN = np.array(FN)
precision = np.zeros(n)
recall = np.zeros(n)
F1 = np.zeros(n)

precision = TP / (TP + FP)
recall = TP/ (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
# # 输出！
# # 终端输出：
# print('{0:<15}{1:<15}{2:<15}{3:<15}'.format('Label', 'Precision', 'Recall', 'F1-score'))
# for i in range(n):
#     print('{0:<15}{1:<15f}{2:<15f}{3:<15f}'.format(list(tagDict.keys())[list(tagDict.values()).index(i)], precision[i], recall[i], F1[i]))

# # 文件输出：
# with open('./output版本1.txt', 'w', encoding='utf-8') as fw:
#     print('{0:<15}{1:<15}{2:<15}{3:<15}'.format('Label', 'Precision', 'Recall', 'F1-score'), file = fw)
#     for i in range(n):
#         print('{0:<15}{1:<15f}{2:<15f}{3:<15f}'.format(list(tagDict.keys())[list(tagDict.values()).index(i)], precision[i], recall[i], F1[i]), file = fw)

file_path = os.path.join('.','output版本1.csv')
with open(file_path, 'w') as f:
    f.write('Label,Precision,Recall,F1-score\n')
    for i in range(n):
        f.write('{0},{1:f},{2:f},{3:f}\n'.format(list(tagDict.keys())[list(tagDict.values()).index(i)], precision[i], recall[i], F1[i]))

data = pd.read_csv(file_path)
print(data.dtypes)