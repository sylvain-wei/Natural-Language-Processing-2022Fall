# 思路
# 第一步，针对每一行line，遍历得到所有的路径，建立有向无环图（具体统计TP的算法需要另外重新实现一遍）
#   每一条边即代表着从一个结点到另一个结点的下标对应的str
# 第二步，对于每一个建立的无环图，进行迪杰斯特拉最短路径算法扫描最短路径，直到到达句子末尾为止。
# TODO:思考如果有多个最短路径如何选取
# 第三步，根据最短路径过程中保存的前序结点，回溯得到结点的序列，reverse一下得到正序的结点序列
#   从结点序列来从前到后遍历一遍，每次将一对结点的对应str放入words中，同时遍历得到intervals
"""
OK1. get_graph(sentence)->graph[list]
OK2. dijkstra(graph)->best_path[list] (调用了get_path)
OK3. get_path(succession, M)->best_path[list](succession表示结点之间的继承关系)
4. least_segment(txt)->words, intervals
5. get_std_intervals()  二维数组存储intervals,每个第0维为一句话的intervals
6. eval_prf()
7. calc_TP()
8. word_cloud()
9. main
10. judge()判断是否分词
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
INF = 10000
dic_words = []
time_tags = ['日', '月', '时', '分']
Chinese_num = ['○', '十', '百', '千', '万', '亿', '％', '０', '－', '点', '分', '之']

def isWord(word):
    '''judge whether the given word can be found in the dictionary'''
    return word in dic_words

def isNum(char):
    '''判断是否是数字'''
    try:
        float(char)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(char)
        return True
    except (TypeError, ValueError):
        pass

    return False

def isAllNum(word):
    for char in word:
        if char in Chinese_num:
            continue
        elif isNum(char):
            continue
        else:
            return False
    return True

def judge(sentence, i, j):
    # 此处的i<=j
    if (isWord(sentence[i:j+1])
        or i == j   # 如果只有单个字了
        or (isAllNum(sentence[i : j])              # 如果是 十九万八千 或 10238 整个串全是数字
            and (((sentence[j] in time_tags) or (sentence[j] == '年' and j - i == 4)) 
                 or isAllNum(sentence[j])))   # 如果是如 10月/23日/23时 或 如果是如 2001年或二○○一年
    ):
        return True
    return False

def get_graph(sentence):
    '''
    input one sentence
    output a graph:
    '''
    # 初始化一个图，是一个点-点矩阵，每一个元素为1或0，表示是否有从一个点到另一个点的有向边
    M = len(sentence)   # M表示节点最大下标，恰好为sentence的长度值（因为从0开始算）
    graph = []
    for i in range(M + 1):
        graph.append([])
        for j in range(M + 1):
            graph[i].append(0)
        if i < M:
            graph[i][i + 1] = 1
    # 扫描整个句子，判断结点之间是否为有向边
    for i in range(M):
        for j in range(i+1, min(i + 23, M + 1)):      # 词最长为22
            if judge(sentence, i, j-1):
                graph[i][j] = 1             # 如果判断需要分词的话，就分出来
    
    return graph

def get_path(succession, M):
    path = [M]
    cur_v = M
    while cur_v != 0:
        path.append(succession[cur_v])
        cur_v = succession[cur_v]
    path = list(reversed(path))
    return path

def dijkstra(graph, max_len):
    '''
    输入一个二维数组表示图, max_len=结点最大下标,是sentence的长度值
    在图中找出一条从0号节点到末尾结点的最短路径
    '''
    M = max_len
    succession = []     # 用一个二维数组存储结点的前继
    found = [0]
    not_found = [i for i in range(1, M+1)]  # 顺序排列
    shortest = np.array([INF for i in range(M+1)])
    shortest[0] = 0                 # 装的各个点到顶点0的探测到的最短距离

    # 初始化
    for i in range(M + 1):
        succession.append(i - 1)   # 初始化，用i-1来作为前继结点 succession[0] = -1
        if graph[0][i] == 1:
            shortest[i] = 1
    # 开始搜索
    while M not in found:
        # 如果从0结点到最后一个结点M的最短路径没有找到的话，不断找最短路径
        # 从没有找到最短路径的点里面找到当前探测的最短距离对应的第一个点
        # v_new_found = list(shortest).index(np.min(shortest[not_found]))
        temp_v_list = np.argwhere(shortest == np.min(shortest[not_found])).reshape(-1)
        for ver in temp_v_list:
            if ver in not_found:
                v_new_found = ver
                break
        # 需要新加入的点从not_found里删除，并加入到found中
        not_found.remove(v_new_found)
        found.append(v_new_found)
        # 找这个新加入的点的邻接节点并判断是否更新最短距离
        # 如果需要更新最短距离，则将这些结点的前继结点进行更新为v_new_found
        for v in not_found:
            if v_new_found == M:    # 找到了M结点的最短路径后就结束了
                break
            if (graph[v_new_found][v] == 1 
                and 1 + shortest[v_new_found] < shortest[v]):
                shortest[v] = shortest[v_new_found] + 1
                succession[v] = v_new_found
    
    return get_path(succession, M)

def get_intervals(path):
    '''将点序列化为路径'''
    len_path = len(path)
    intervals = []
    for i in range(len_path - 1):
        left = path[i]
        right = path[i+1] - 1
        intervals.append((left, right))
    return intervals

def least_segment(txt):
    '''
    传入一个文本文件
    按行进行get_graph 并 dijkstra 来get_path
        然后用path来get_intervals作为这一行的intervals
    '''
    line_counter = 0
    prd_intervals = []
    prd_words = []
    for entry in txt:
        sentence = entry.strip()
        graph = get_graph(sentence)
        path = dijkstra(graph, len(sentence))
        line_intervals = get_intervals(path)
        prd_intervals.append(line_intervals)
        for (left, right) in line_intervals:
            prd_words.append(sentence[left: right+1])
        line_counter += 1
    
    return (prd_words, prd_intervals)

def get_std_intervals():
    std_intervals = []  # 二维数组，每一行存一行句子的区间
    with open('../分词对比文件/gold.txt', 'r', encoding='utf-8') as fr:
        for entry in fr:
            words = entry.split()
            line_intervals = []
            left = 0
            for word in words:
                right = len(word) - 1 + left
                line_intervals.append((left, right))
                left = right + 1
            std_intervals.append(line_intervals)
    return std_intervals

def calc_TP(std_intervals, prd_intervals):
    TP = 0
    for i in range(len(std_intervals)):
        TP += len(set(std_intervals[i]) & set(prd_intervals[i]))
    return TP

def size_2darray(arr):
    count = 0
    for i in range(len(arr)):
        count += len(arr[i])
    return count

def eval_PRF1(std_intervals, prd_intervals):
    TPaddFN = size_2darray(std_intervals)
    TPaddFP = size_2darray(prd_intervals)
    TP = calc_TP(std_intervals, prd_intervals)
    precision = TP / TPaddFP
    recall = TP / TPaddFN
    F1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, F1)

def word_cloud():
    stop_words = [line.strip() for line in open('../停用词/stop_word.txt', 'r', encoding='utf-8')]
    # 获取想要得到的
    with open('../喜欢的文章段落/《铸剑》by鲁迅.txt', 'r', encoding='utf-8') as f:
        words, __ = least_segment(f) # 将str格式用BM算法来获取分词结果
        # 去除停用词，得到sentences
        sentences = ""
        for word in words:
            if word in stop_words:
                continue
            sentences += str(word)+' '
        # 生成词云图
        cloud_img = WordCloud(background_color='white',
                              font_path='../字体/plun04wkyso54jx5893el7t1zkz94ix1.otf',
                              width=2000,
                              height=2000,).generate(sentences)
        plt.imshow(cloud_img)
        plt.axis('off')
        plt.savefig('../词云/《铸剑》by鲁迅_最小分词法')
        plt.show()

# main
# 首先通过对待分词文件进行算法预测，计算分词对比文件

with open('../词典/pku_training_words.utf8', 'r', encoding='utf8') as dic:
    max_len = 0
    for entry in dic:
        dic_words.append(entry.strip())
        if max_len < len(entry.strip()):
            max_len = len(entry.strip())

with open('../待分词文件/corpus.txt', 'r', encoding='utf-8') as fr:
    res_words, res_intervals = least_segment(fr)
    std_intervals = get_std_intervals()
    P, R, F1 = eval_PRF1(std_intervals, res_intervals)
    # if ~os.path.exists(os.path.join('.','评估结果')):
    #     os.mkdir(os.path.join('.','评估结果'))
    eval_res = os.path.join('.','评估结果','eval_res_leastSegment.txt')
    with open(eval_res, 'w', encoding='utf-8') as fw:
        content = 'Presicion:{0:f}\nRecall:{1:f}\nF1:{2:f}'.format(P, R, F1)
        fw.write(content)
        print(content)

# 最后生成一个词云
word_cloud()