'''
算法API:
1.正向最大匹配FMM():输入是一个str,输出是一个数组元素,下标从小到大是该句子中从前到后的划分结果
2.逆向最大匹配BMM()
3.BM()总算法,调用FMM和BMM,进行比较,输入是txt所有文字,输出是基于文章的分词结果数组(反正是一个迭代器)
其中,双向匹配的原则:选取FMM和BMM分出来的词的个数中较小的那个结果.如果分出来词数结果相同,则选择去重之后,词数最少的那个结果
4.main代码:根据BM()迭代器,实现空格划分,输出结果
5.词云图生成模块(一个文件的读取操作)word_cloud(),其中需要去除停用词
6.查词典的模块IsWord(),读取pku_training_words.utf8字典查取是否需要分词,返回值为True|False
7.模型评估的三个指标TP的计算模块 calc_TP()
8.三个指标Precision、Recall、F1的评估模块eval_PRF1()
9.规则匹配算法模块,将日期等固有形式的词组匹配出来match_date_num()TODO:
'''
# TODO:要考虑未登录词、日期词等的处理
# TODO:这里需要先将句子进行切分，然后对每一个句子进行判断
'''最大分词中包含了停用词'''

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import torch
# 为了避免每次都从文件中读取，降低速度，这里使用dic全局数组来存储所有的字典内容
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

def FMM(entry_list):
    """
    Through FMM, get the partition array(type:list) for words in the range given.
    param: string:待分词的句子
    return: 分词的结果FMM_res(list) 
    """
    # 初始化FMM中存词典的词
    FMM_res = []
    # 为了后续count区间，所以需要对所有的FMM预测区间进行存储
    FMM_intervals = []
    sentence_begin = 0
    line_num = 0
    for entry in entry_list:   # TODO:这样做是有点问题的，可以想一下如何拆分句子以达到最优的拆分方式。是否可以考虑每22个字的句子拆分一次呢？从前往后地
        # 那在BMM的时候是否就可以整个段落句子从后往前，然后处理
        line_num += 1
        sentence = entry.strip()
        N = len(sentence)
        i = 0
        while i < N:    #注意：当用for i in range(N)的时候，每次是从一个迭代器里面查找的
            for j in range(min(N-1, i+21), i-1, -1):
                if j-i+1 > 22:
                    continue
                # if i == j and isWord(sentence[i]) == False:
                #     FMM_res.append(sentence[i])
                #     FMM_intervals.append((i, i))
                # if isWord(sentence[i:j+1]) or i == j:
                    # i==j表示可能这个词典中并没有找到这个词，但是也需要划分出去！
                if judge(sentence, i, j):   # 判断是否切分
                    FMM_res.append(sentence[i:j+1])
                    FMM_intervals.append((sentence_begin + i, sentence_begin + j))
                    i = j
                    break
            i += 1
        # TODO:处理一下细节问题，比如‘母亲’ ‘亲’都出现了。。。这个咋解决？
        sentence_begin += N
    return (FMM_res, FMM_intervals)

def BMM(entry_list):# TODO:更改每一次循环的下标，保证intervals中的下标是正确的
    BMM_res = []
    BMM_intervals = []
    sentence_begin = 0
    line_num = 0
    for entry in entry_list:
        line_num += 1
        sentence = entry.strip()
        N = len(sentence)
        i = N - 1
        while i >= 0:
            for j in range(max(0, i-21), i+1):
                if i-j+1 > 22:
                    continue
                # if isWord(sentence[j:i+1]) or i == j:
                if judge(sentence, j, i):   # 判断是否拆分
                    BMM_res.append(sentence[j:i+1])
                    BMM_intervals.append((sentence_begin + j, sentence_begin + i))
                    i = j
                    break
            i -= 1
        sentence_begin += N
    return (list(reversed(BMM_res)), list(reversed(BMM_intervals)))

def BM(txt):
    """
    输入为一个文章段落,首先对文章进行预处理操作TODO:
    传入txt文件变量即可
    res表示BM的结果
    下面选取两个算法得到的结果中最好的结果
    flag: 如果为0,返回FMM,如果为1,返回BMM
    """
    entry_list = []
    for entry in txt:
        entry_list.append(entry.strip())
    FMM_res, FMM_intervals = FMM(entry_list)
    BMM_res, BMM_intervals = BMM(entry_list)
    if len(FMM_res) < len(BMM_res):
        flag = 0
    elif len(FMM_res) > len(BMM_res):
        flag = 1
    else:
        # 去重，取去重之后结果最小的那个作为结果
        if len(list(dict.fromkeys(FMM_res))) < len(list(dict.fromkeys(BMM_res))):
            flag = 0
        else:
            flag = 1
    if flag == 0:
        return (FMM_res, FMM_intervals)
    else:
        return (BMM_res, BMM_intervals)

def word_cloud():
    stop_words = [line.strip() for line in open('../停用词/stop_word.txt', 'r', encoding='utf-8')]
    # 获取想要得到的
    with open('../喜欢的文章段落/《铸剑》by鲁迅.txt', 'r', encoding='utf-8') as f:
        words, __ = BM(f) # 将str格式用BM算法来获取分词结果
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
        plt.savefig('../词云/《铸剑》by鲁迅_最大匹配法')
        plt.show()

def calc_TP(std_intervals, prd_intervals):
    '''
    计算TP
        在这里需要首先定义一下Positive是什么:
        Positive即 预测分词区间在标准分词结果区间中的个数
        Negative即 预测分词区间不在标准分词结果区间中的个数
        标准分词结果中的所有区间的总数即为TP+FN
        预测分词结果中的所有区间的总数即为TP+FP
        预测分词结果中的所有区间与标准分词结果中的所有区间的交集大小即为TP
    '''
    TP = 0
    for interval in prd_intervals:
        if interval in std_intervals:
            TP += 1
    return TP


def eval_PRF1(std_intervals, prd_intervals):
    TPaddFN = len(std_intervals)
    TPaddFP = len(prd_intervals)
    TP = calc_TP(std_intervals, prd_intervals)
    precision = TP / TPaddFP
    recall = TP / TPaddFN
    F1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, F1)

def get_std_intervals():
    with open('../分词对比文件/gold.txt', 'r', encoding='utf-8') as fr:
        std_string = ''
        for entry in fr:
            std_string = std_string + entry.strip('\n')
        len_passage = len(std_string)
        std_intervals = []
        left = 0
        j = -1
        i = 0
        std_words = []
        while i < len_passage:
            if i+2 < len_passage and std_string[i+1] == ' ' and std_string[i+2] == ' ':
                len_string = i - j
                std_intervals.append((left, left + len_string - 1))
                std_words.append(std_string[j+1: i+1])
                left = left + len_string
                i = i + 2
                j = i
            i += 1
        std_words = std_words
        return std_intervals

# main
# 首先通过对待分词文件进行算法预测，计算分词对比文件

with open('../词典/pku_training_words.utf8', 'r', encoding='utf8') as dic:
    max_len = 0
    for entry in dic:
        dic_words.append(entry.strip())
        if max_len < len(entry.strip()):
            max_len = len(entry.strip())

with open('../待分词文件/corpus.txt', 'r', encoding='utf-8') as fr:
    res_words, res_intervals = BM(fr)
    std_intervals = get_std_intervals()
    P, R, F1 = eval_PRF1(std_intervals, res_intervals)
    # if ~os.path.exists(os.path.join('.','评估结果')):
    #     os.mkdir(os.path.join('.','评估结果'))
    eval_res = os.path.join('.','评估结果','eval_res_BM.txt')
    with open(eval_res, 'w', encoding='utf-8') as fw:
        content = 'Presicion:{0:f}\nRecall:{1:f}\nF1:{2:f}'.format(P, R, F1)
        fw.write(content)
        print(content)
# 最后生成一个词云
word_cloud()