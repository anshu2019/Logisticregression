
# Import the required librray
import sys
import numpy as np
import re
import csv
import json
eps = np.finfo(float).eps
from numpy import log2 as log
import time


start = time.time()

#commandline param
path1= sys.argv[1]
path2= sys.argv[2]
path3= sys.argv[3]
path4= sys.argv[4]
path5= sys.argv[5]
path6= sys.argv[6]
path7= sys.argv[7]
feature_flag=int(sys.argv[8])

#variables
trn_data=[]
val_data=[]
tes_data=[]
dictionary={}
dict_len=0

           
#function for model 1         
def create_model1_features(label,review):
    #get unique value    
    raw= review.split(" ")
    unique = []
    [unique.append(item) for item in raw if item not in unique]
    #write label
    fileStr=label
    for itm in  unique:
        if itm in dictionary : 
             indx=dictionary[itm].strip()
             fileStr+='\t'+str(indx)+':1'
    fileStr = fileStr.strip('\t')
    return fileStr

#function for model 2        
def create_model2_features(label,review):
    #get unique value    
    raw= review.split(" ")
    unique = []
    [unique.append(item) for item in raw if item not in unique]
    
    #write label
    fileStr=label
    for itm in  unique:
        if(raw.count(itm)<4):
            if itm in dictionary : 
                indx=dictionary[itm].strip()
                fileStr+='\t'+str(indx)+':1'
        
    fileStr = fileStr.rstrip('\t')
    return fileStr

# model switch
def model_switch(label,review,flag):
    if(flag==1):
        fileStr=create_model1_features(label,review)
    elif(flag==2):
        fileStr=create_model2_features(label,review)
    return fileStr


#create dictionary           
with open(path4, 'r') as tsv1:                     
            for r in tsv1:
                dt = r.split(" ")
                dictionary[dt[0]]=dt[1]
                
# create feature for train data
with open(path1, 'r') as tsv2:
    with open(path5, "w") as out_train:
              for r in tsv2:
                data = r.split("\t")                
                fileStr=model_switch(data[0],data[1],feature_flag)            
                out_train.writelines(fileStr+'\n')
    out_train.close()         

#create feature for validation data            
with open(path2, 'r') as tsv2:
    with open(path6, "w") as out_valid:
              for r in tsv2:
                data = r.split("\t")                
                fileStr=model_switch(data[0],data[1],feature_flag)            
                out_valid.write(fileStr+'\n')
    out_valid.close() 

#create feature for test data   
with open(path3, 'r') as tsv2:
    with open(path7, "w") as out_test:
              for r in tsv2:
                data = r.split("\t")                
                fileStr=model_switch(data[0],data[1],feature_flag)            
                out_test.write(fileStr+'\n')
    out_test.close()  
            
            

# run your code
end = time.time()

elapsed = end - start
print("done in - %f" %(elapsed)) 

