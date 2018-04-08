# @Time    : 2018/4/8 15:39
# @Author  : 王江涛
# @File    : Bayes.py
# @Software: PyCharm
# ########################################################
# 朴素贝叶斯的一般过程
# 收集数据：可以使用任何方式
# 准备数据：需要数据型或是布尔型数据
# 分类数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好
# 训练算法：计算不同的独立特征的条件概率
# 测试算法：计算错误率
# 使用算法：文档分类
# ########################################################
#
#
# 导入基础模块
import os
import sys
from numpy import *

# 加载数据集
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

# 根据样本创建一个词库
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

# 统计每个样本在词库中的出现情况
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 计算条件概率和类标签概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs) #计算某个类发生的概率
    p0Num = ones(numWords); p1Num = ones(numWords) #初始样本个数为1，防止条件概率为0，影响结果
    p0Denom = 2.0; p1Denom = 2.0  #作用同上
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)         #计算类标签为1时的其它属性发生的条件概率
    p0Vect = log(p0Num/p0Denom)         #计算标签为0时的其它属性发生的条件概率
    return p0Vect,p1Vect,pAbusive       #返回条件概率和类标签为1的概率

# 训练贝叶斯分类算法
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 文档词袋模型,修改函数setOfWords2Vec
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 测试函数
def testingNB():
    #step1：加载数据集和类标号
    listOPosts,listClasses = loadDataSet()
    #step2：创建词库
    myVocabList = createVocabList(listOPosts)
    # step3：计算每个样本在词库中的出现情况
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #step4：调用第四步函数，计算条件概率
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    # step5
    # 测试1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    # 测试2
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
# 实验
if __name__=='__main__':
    testingNB()

