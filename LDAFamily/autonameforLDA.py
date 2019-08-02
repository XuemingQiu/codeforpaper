# _*_coding: utf-8_*_
"""
    @describe:  实现论文 Automatic_Labeling_of_Topic_Models_Using_Text_Summaries.pdf
                基于子模优化的文本主题自动命名，贪婪的抽取文章中的句子做为lda主题模型跑出来的抽象结果进行命名
    @author: xuemingQiu
    @date 19-6-13
"""
import csv
import math
import random
import re
<<<<<<< HEAD
from datetime import datetime
=======
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

jieba.load_userdict("docs/lexicon.txt")

__all__ = ['GetLDA', 'CsvOp', 'PreDealData', 'ExteactSummary']

<<<<<<< HEAD
# 模型参数名称和位置
tf_model = "model/tfmodel"
lda_model = "model/ldamodel"
lda_result_topic = "ldaresult/topics.txt"
lda_result_probility = "ldaresult/probility.txt"
# LDA参数表
alpha = 0.1  # 超参数
n_topics = 25  # 主题个数
beta = 0.01  # 超参数
max_iters = 5  # 迭代轮数
n_top_wpords = 50  # 每个主题下的top-n的单词数
# 词频特征筛选
max_df = 0.95  # 最大的频率值
min_df = 2  # 最小的逆频率书
max_features = 10000  # 最大的单词个数


class GetLDA:
    def __init__(self, corpus=None):
        self.alpha = alpha
        self.n_topics = n_topics
        self.beta = beta
        self.corpus = corpus
        self.max_iters = max_iters
        self.n_top_wpords = n_top_wpords
        
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
    
    def train(self):
        try:
            tf_vectorize = joblib.load(tf_model)
            tf = tf_vectorize.fit_transform(self.corpus)
        except:
            tf_vectorize = CountVectorizer(max_df=self.max_df, min_df=self.min_df, max_features=self.max_features)
            tf = tf_vectorize.fit_transform(self.corpus)
            joblib.dump(tf_vectorize, tf_model)
        print("complete training tfidf........")
        try:
            ldamodel = joblib.load(lda_model)
            # ldamodel.fit(tf)
        except:
            ldamodel = LatentDirichletAllocation(n_components=self.n_topics, max_iter=self.max_iters, n_jobs=-1,
                                                 learning_method='batch')
            ldamodel.fit(tf)
            joblib.dump(ldamodel, lda_model)
=======

class GetLDA:
    def __init__(self, corpus=None):
        self.alpha = 0.1
        self.n_topics = 25
        self.beta = 0.01
        self.corpus = corpus
        self.max_iters = 100
        self.n_top_wpords = 50
    
    def train(self):
        try:
            tf_vectorize = joblib.load("model/tfmodel")
            tf = tf_vectorize.fit_transform(self.corpus)
        except:
            tf_vectorize = CountVectorizer(max_df=0.95, min_df=2, max_features=10000)
            tf = tf_vectorize.fit_transform(self.corpus)
            joblib.dump(tf_vectorize, "model/tfmodel")
        print("complete training tfidf........")
        try:
            ldamodel = joblib.load("model/ldamodel")
            # ldamodel.fit(tf)
        except:
            ldamodel = LatentDirichletAllocation(n_components=self.n_topics, max_iter=self.max_iters,
                                                 learning_method='batch')
            ldamodel.fit(tf)
            joblib.dump(ldamodel, "model/ldamodel")
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
        print("complete training lda !! ")
        n_top_words = self.n_top_wpords
        tf_fetures_name = tf_vectorize.get_feature_names()
        self.get_top_words(ldamodel, tf_fetures_name, n_top_words)
    
    def get_top_words(self, model, feature_names, n_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            print("topic index %d" % topic_idx)
            print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
<<<<<<< HEAD
        with open(lda_result_topic, "w") as f:
=======
        with open("ldaresult/topics.txt", "w") as f:
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
            id = 0
            for topic in topics:
                f.write("Topic " + str(id) + " " + topic + "\n")
                id += 1
        print(model.components_)
        topics_p = pd.DataFrame(model.components_)
<<<<<<< HEAD
        topics_p.to_csv(lda_result_probility, index=False, header=0)
=======
        topics_p.to_csv("ldaresult/probility.txt", index=False, header=0)
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
    
    def get_p_of_topic_word(self):
        """
        :return: [[p_a,p_b,p_c,...],[]]
        """
        # 获取概率
        p = []
<<<<<<< HEAD
        with open(lda_result_probility, "r") as f:
=======
        with open("ldaresult/probility.txt", "r") as f:
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
            for line in f.readlines():
                temp = line.strip().split(",")
                temp = [float(i) for i in temp]
                p.append(temp)
        return p
    
    def get_topic_words(self):
        """
        :return:[[a,b,c...],[]]
        """
        # 获取主题下的单词
        theta_list = []
<<<<<<< HEAD
        with open(lda_result_topic, "r") as f:
=======
        with open("ldaresult/topics.txt", "r") as f:
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
            for line in f.readlines():
                theta_list.append(line.strip().split()[2:])
        return theta_list


class CsvOp:
    """
    1. 用来读去csv文件
    """
    
    def __init__(self):
        pass
    
    def csvReader(self, filePath):
        """
        :param filePath: 文件路径
        :return: 返回[[a,b,c...],[c,d,e..],[]]
        """
        dataset = []
        try:
            f = open(filePath)
            for line in csv.reader(f):
                dataset.append(line)
        except:
            print("the file is not exist or the file's content is wrong!")
        return dataset


class PreDealData:
    def __init__(self):
        pass
    
    def read_origin_data(self):
<<<<<<< HEAD
        """
        读取原来的case_info-456的数据
        :return:  title+content的内容，保存成csv
        """
=======
        '''
        读取原来的case_info-456的数据
        :return:  title+content的内容，保存成csv
        '''
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
        print("starting reading origin data ~~~")
        filename = "docs/case_info-456.csv"
        csvop = CsvOp()
        data = csvop.csvReader(filename)
        data = pd.DataFrame(data[1:],
                            columns=['title', 'content'
                                     ])
        data = pd.DataFrame(data['CASE_TITLE'] + "。" + data['CASE_CONTENT'], columns=['content'])
        data.to_csv("data/origin.csv", index=False)
    
    def get_stop_word(self):
        print("starting reading stopword data ~~~")
<<<<<<< HEAD
        """
        :return: 停用词表
        """
=======
        '''
        :return: 停用词表
        '''
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
        path = "docs/stopword.txt"
        stopwords = []
        with open(path) as f:
            for line in f.readlines():
                stopwords.append(line.strip())
        return stopwords
    
    def cutword(self, setence_list):
        
<<<<<<< HEAD
        """
        :param setence: 句子列表
        :return: 分词结果
        """
=======
        '''
        :param setence: 句子列表
        :return: 分词结果
        '''
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
        print("starting cut setence to word lists ~~~")
        stopword = self.get_stop_word()
        setences = []
        for i in setence_list:
            setences.append(" ".join([i for i in list(jieba.cut("。".join(i.split()))) if i not in stopword]))
        return setences
    
    def split_setentce_to_words(self):
<<<<<<< HEAD
        """
        对原来的数据进行分词结果
        :return: 保存分词后的结果
        """
=======
        '''
        对原来的数据进行分词结果
        :return: 保存分词后的结果
        '''
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
        print("starting read setence to cut  ~~~")
        self.read_origin_data()
        data2 = pd.read_csv("data/origin.csv", header=0)
        data2 = pd.DataFrame(self.cutword(data2['content']), columns=["cutword"])
        data2.to_csv("data/cutword.csv", index=False)
        return data2
    
    def get_cuted_setence(self):
        print("starting get cutted setence ~~~")
        corpus = []
        filename = "data/cutword.csv"
        data = pd.read_csv(filename, header=0)
        for line in data['cutword']:
            # print(line)
            corpus.append(line.strip())
        return corpus


class ExteactSummary:
    def __init__(self):
        pass
    
<<<<<<< HEAD
    def computeKL(self, p, TW, SW, s):
        """
        :param p: 单词w在每个主题theta下的分布概率
=======
    # 计算主题摘要
    def computeKL(self, p, TW, SW, s):
        """
        :param p: 单词w在每个主题theta下的分布概率
        :param theta: 主题
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
        :param TW: 主题theta下的top 500单词集
        :param SW: 句子s移除停用词后的的集合
        :param s: 就是句子s
        :return: kl散度
        """
        KL = 0
        for w, sw in TW, SW:
            tf_w_s = 0  # 单词w在句子中出现的次数。
            for i in s:
                if i == w:
                    tf_w_s += 1
            if tf_w_s == 0:
                KL += p[w] * math.log(p[w] / 0.00001)
            else:
                KL += p[w] * math.log(p[w] / (tf_w_s / len(s)))
        return KL
    
    def getSummaryExtraction(self, V, p_theta, theta_list, theta_index, TW):
        """
        :param V:  句子集V
        :param p_theta: 每个主题下每个单词的概率
        :param theta_list: 主题-单词 表
        :return: 摘要的句子
        """
        E, U = [], V
        alpha = 0.05
        beta = 250
        gama = 300
        epsion = 0.15
        L = 250  # 句子长度
        V_cut = {}
        data = pd.read_csv("data/cutword.csv", header=0)
        for line, wordlist in zip(V, data['cutword']):
            V_cut[line] = wordlist.split(" ")
        
        def sim(s1, s2):
            """
            :param s1: 句子1
            :param s2: 句子2
            :return: 返回句子的余弦相似度
            """
            # 分词
            cut1 = V_cut[s1]
            cut2 = V_cut[s2]
            list_word1 = (','.join(cut1)).split(',')
            list_word2 = (','.join(cut2)).split(',')
            
            # 列出所有的词,取并集
            key_word = list(set(list_word1 + list_word2))
            # 给定形状和类型的用0填充的矩阵存储向量
            word_vector1 = np.zeros(len(key_word))
            word_vector2 = np.zeros(len(key_word))
            
            # 计算词频
            # 依次确定向量的每个位置的值
            for i in range(len(key_word)):
                # 遍历key_word中每个词在句子中的出现次数
                for j in range(len(list_word1)):
                    if key_word[i] == list_word1[j]:
                        word_vector1[i] += 1
                for k in range(len(list_word2)):
                    if key_word[i] == list_word2[k]:
                        word_vector2[i] += 1
            
            # 输出向量
            dist1 = float(
                np.dot(word_vector1, word_vector2) / (np.linalg.norm(word_vector1) * np.linalg.norm(word_vector2)))
            return dist1
        
        def REL_E(E2):
<<<<<<< HEAD
            """
            :return: REL(E)
            """
=======
            '''
            :return: REL(E)
            '''
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
            rel_e = 0
            for s_dot in V:
                SUM_E = 0
                for s in E2:
                    SUM_E += sim(s_dot, s)
                sum_v = 0
                for s in V:
                    sum_v += sim(s_dot, s)
                rel_e += min(SUM_E, alpha * sum_v)
            return rel_e
        
        def COV_E(E2):
            cov_e = 0
            for w in TW:
                tf_w_s = 0
                for s in E2:
                    for i in V_cut[s]:
                        if i == w:
                            tf_w_s += 1
                cov_e += (p_theta[theta_index][w] * math.sqrt(tf_w_s))
            cov_e *= beta
            return cov_e
        
        def DIS_E(E2):
            dis_e = 0
            for theta in range(len(theta_list)):
                for s in E2:
                    for w in TW:
                        tf_w_s = 0
                        for i in V_cut[s]:
                            if i == w:
                                tf_w_s += 1
                        # print(theta, " ** " , p_theta[theta])
                        if w in p_theta[theta].keys():
                            dis_e += p_theta[theta][w] * tf_w_s
            dis_e *= (-gama) * dis_e
            return dis_e
        
        print("starting ~~~~~~~`")
        sum_s = 0
        while len(U) != 0:
            s_hat = ""
            max_s_hat = float("-inf")
            print("U length = ", len(U))
            temp = E[:]
            f_e = REL_E(temp) + COV_E(temp) + DIS_E(temp)
            for s in U:
                SW = V_cut[s]
                # print("句子 = ", s, "    分词 = ", SW)
                temp.append(s)
                f_e_hat = REL_E(temp) + COV_E(temp) + DIS_E(temp)
                # print("f_e_hat = ", f_e_hat, " ,  f_e = ", f_e)
                if (f_e_hat - f_e) / math.pow(len(SW), epsion) > max_s_hat:
                    max_s_hat = (f_e_hat - f_e) / math.pow(len(SW), epsion)
                    s_hat = s
            print("select = ", s_hat)
            if sum_s + len(V_cut[s_hat]) <= L and max_s_hat * math.pow(len(V_cut[s_hat]), epsion) >= 0:
                sum_s += len(V_cut[s_hat])
                print("append to E: = ", s_hat)
                E.append(s_hat)
            U.remove(s_hat)
        
        return E
    
    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")


def getlda():
<<<<<<< HEAD
    """
    lda training function
    :return:
    """
=======
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
    predata = PreDealData()
    corpus = predata.get_cuted_setence()
    print("corpus length  = ", len(corpus))
    lda = GetLDA(corpus)
    lda.train()


def getExtract():
    ldamodel = GetLDA()
    # 获取主题-单词
    theta_list = ldamodel.get_topic_words()
    # 获取主题下每个单词的概率
    p = ldamodel.get_p_of_topic_word()
    # 对应每个主题下每个单词的
    p_theta = []
    
    row = 0
    for topic in theta_list:
        temp = {}
        colum = 0
        for w in topic:
            temp[w] = p[row][colum]
            colum += 1
        p_theta.append(temp)
        row += 1
    
    print("topic nums = ", len(p_theta))
    print("p_theta 0:", p_theta[0])
    # 读取原始的title
    filename = "docs/case_info-456.csv"
    csvop = CsvOp()
    data = csvop.csvReader(filename)
    V = [i[1] for i in data[1:]]
    print("V length = ", len(V))
    print("V[0-10] = ", V[:10])
    
    get_cut_title = []
    with open("data/cut_title.csv", "r") as f:
        for line in f.readlines()[1:]:
            get_cut_title.append(line)
    print("get cuting title length = ", len(get_cut_title))
    if len(get_cut_title) == 0:
        get_cut_title = PreDealData().cutword(V)
        result_title = pd.DataFrame(get_cut_title, columns=['title'])
        result_title.to_csv("data/cut_title.csv", index=False)
    
    for i in range(len(theta_list)):
        topic_V = []
        for title in range(len(get_cut_title)):
            for w in get_cut_title[title].split():
                if w in theta_list[i]:
                    topic_V.append(V[title])
                    break
        print("candidate setences of topic %d" % i, " = ", topic_V)
        TW = theta_list[i]
        print("topic %d : " % i, TW)
<<<<<<< HEAD
        E = ExteactSummary().getSummaryExtraction(random.sample(topic_V, 30), p_theta, theta_list, i, TW)
=======
        E = ExteactSummary().getSummaryExtraction(random.sample(topic_V, 100), p_theta, theta_list, i, TW)
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
        print("*" * 200)
        print("select setence = ", E)
        print("select setence length = ", len(E))
        with open("result.txt", "a+") as f:
            f.write("Topic " + str(i) + " : ")
            f.write(" ".join(E))
            f.write("\n")
    print("#" * 100)


def main():
    # predata = PreDealData()
    # predata.split_setentce_to_words()
<<<<<<< HEAD
    start_time = datetime.now()
    print("strat time = ", start_time)
    getlda()
    end_time = datetime.now()
    print("strat time = ", end_time)
    print("all time cost = ", end_time - start_time)
=======
    # getlda()
>>>>>>> e949ed10e195d8518d212a9f5334a3f0629e4e66
    getExtract()


if __name__ == '__main__':
    main()
