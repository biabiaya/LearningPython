from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inputX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] # 计算矩阵的行数
    diffMat = tile(inputX,(dataSetSize,1)) - dataSet # 计算差值
    sqDiffMat = diffMat**2 # 计算欧式距离（两点间的距离公式）sq=sqrt
    sqDistances = sqDiffMat.sum(axis=1) # 特征值平方相加
    distance = sqDistances**0.5 # 计算欧式距离（两点间的距离公式）sq=sqrt
    sortedDistIndicies = distance.argsort()
    classCount={}
    for i in range(k):
        index = sortedDistIndicies[i] # 获取第i小的值的下标值
        voteIlabel = labels[index] # 获取第i小的值的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1 # 分类计数

    result = sorted(classCount.items(),key = lambda item:item[1])

    return result[0][0]


# 调用    
g,l = createDataSet()

print( classify0([0,0.1],g,l,1) )


