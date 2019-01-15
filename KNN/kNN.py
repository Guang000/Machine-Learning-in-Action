
# coding: utf-8

# In[1]:


import numpy as np
import operator
from os import listdir


# In[2]:


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# In[3]:


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet #tile函数对inX进行复制
    sqDiffMat = np.square(diffMat)
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = np.sqrt(sqDistances)
    sortedDistIndicies = np.argsort(distances) #argsort函数将distances从小到大排列并提取索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #字典get函数返回指定键的值，不存在则返回第二个参数默认值
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #按classCount的第二属性从大到小排序
    return sortedClassCount[0][0]


# In[4]:


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFormLine = line.split('\t')
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector


# In[5]:


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# In[6]:


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio) #10%测试数据
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("The total error rate is: %f" % float(errorCount / numTestVecs))


# In[7]:


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("Percentage of time spent playing video games?"))
    ffMiles = float(input("Frequent flier miles earned per year?"))
    iceCream = float(input("Liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([percentTats, ffMiles, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])


# In[8]:


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


# In[9]:


def handwritingClassTest():
    hwLabels = []
    traingFileList = listdir('trainingDigits')
    m = len(traingFileList)
    traingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        traingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, traingMat, hwLabels, 3)
        print("The classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
    if(classifierResult != classNumStr):
        errorCount += 1
    print("\nThe total number of errors is: %d" % errorCount)
    print("\nThe total error rate is: %f" % (errorCount / float(mTest)))
        

