# Import the required librray
import sys
import numpy as np
import re
import csv
import math
eps = np.finfo(float).eps
from numpy import log2 as log
from math import exp
from csv import reader
import time

# time start........................................................................
start = time.time()

#commandline param
path1= sys.argv[1]
path2= sys.argv[2]
path3= sys.argv[3]
path4= sys.argv[4]
path5= sys.argv[5]
path6= sys.argv[6]
path7= sys.argv[7]
no_epoch= int(sys.argv[8])

l_rate=0.1
dict_len=0

# Load a file
def load_dataFile(filepath):
    dataset = list()
    with open(filepath, 'r') as file:
        csv_reader = reader(file, delimiter='\t')
        for row in csv_reader:
            if not row:
                   continue
            dataset.append(row)
    return dataset

# Make string data from files as float
def str_to_float(lis):
    return [[float(j.strip()) for j in i] for i in lis]

#prepare data for calculation
def normalizelist(lis):
    return [[j.replace(":1","") for j in i] for i in lis]
              
       

# sigmoid function
def sigmoid_func(one_row, theeta):    
    yhat = theeta[0]*1 # this theeta0 =1 , taking account for bias value x0=1
    for i in range(1,len(one_row)):
        yhat += theeta[int(one_row[i])+1] * 1
    return 1.0 / (1.0 + exp(-yhat))        


#negative log likelyhood function
def get_neg_loglikelyhood(data,theeta): 
    j_theeta= 0
    for d in data:
        one_row = d        
        yhat=theeta[0]*1
        for k in range(1,len(one_row)-1):
            yhat += theeta[int(one_row[k])+1] * 1
        
        j_theeta+= -float(d[0])*yhat + math.log(1 + exp(yhat))
    
    return j_theeta
    
# apply stochastic gradient descent to get theeta value
def get_theeta_by_sgd(train,validndata):
    theeta = [0.0 for i in range(dict_len+1)]
    theeta[0] =0 # this theeta0 =0 , taking account for bias value x0=1
    for epoch in range(no_epoch):
        for one_row in train:
            yhat = sigmoid_func(one_row, theeta)
            error = one_row[0] - yhat
            theeta[0] = theeta[0] + l_rate * error * 1
            for i in range(1,len(one_row)):
                theeta[int(one_row[i])+1] = theeta[int(one_row[i])+1] + l_rate * error * 1
        #claculate negative log likelyhood on validation data.
        #neg_log = get_neg_loglikelyhood(validndata,theeta)
        #neg_log = float(neg_log)/len(validndata)
        #print("loglikelyhood %f" %(neg_log))
    return theeta



# apply logistic regression
def test_logis_reg(test,theeta,file):
    predictions = list()    
    for row in test:
        yhat = sigmoid_func(row, theeta)
        if(yhat >=0.50):
             yhat = 1
        else:
             yhat = 0
       
        predictions.append(yhat)
        file.writelines(str(yhat)+'\n')
    return(predictions)

# evaluate accuracy
def get_accuracy(actual, predicted):
    total =len(actual)
    correct = 0
    for i in range(len(actual)):
        if actual[i][0] == predicted[i]:
            correct += 1
    return str.format('{0:.6f}', float(total-correct) / float(total+eps))



# Test the Logistic regression....
errMetricFile = open(path7,"w")
trnOutLbl = open(path5,"w")
tesOutLbl = open(path6,"w")

#training data
traindata = load_dataFile(path1)
traindata =normalizelist(traindata)
traindata = str_to_float(traindata)

#dictionary data
dictionary =load_dataFile(path4)
dict_len = len(dictionary)

#validation data
validndata = load_dataFile(path2)
validndata =normalizelist(validndata)
validndata = str_to_float(validndata)

#test data
testdata = load_dataFile(path3)
testdata =normalizelist(testdata)
testdata = str_to_float(testdata)

#apply regression
theeta=get_theeta_by_sgd(traindata,validndata)

#test training data
result1 =test_logis_reg(traindata,theeta,trnOutLbl)
errtrn =get_accuracy(traindata,result1)
print(errtrn)
#test test data
result2 =test_logis_reg(testdata,theeta,tesOutLbl)
errtes =get_accuracy(testdata,result2)
print(errtes)

#error logging

errMetricFile.writelines("error(train): "+str(errtrn)+"\n")
errMetricFile.writelines("error(test): "+str(errtes)+"\n")
errMetricFile.close()
trnOutLbl.close()
tesOutLbl.close()

# code ends here.................................................................
end = time.time()
elapsed = end - start
print("done in - %f" %(elapsed))