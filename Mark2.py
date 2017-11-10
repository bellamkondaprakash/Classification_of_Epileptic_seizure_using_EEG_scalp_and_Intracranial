
# coding: utf-8

# # import all depedencies

# In[21]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import os
import pyeeg
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 



# # Load files from folders 

# In[22]:

# import each and every file

dirB="/home/putus/PROJECT/EEG/setB/"
tempB = []
for file in os.listdir(dirB):
    fl = dirB + file
    tempB.append(fl)
tempB = sorted(tempB)    # class: 1     val:  -1

dirC="/home/putus/PROJECT/EEG/setC/"
tempC = []
for file in os.listdir(dirC):
    fl = dirC + file
    tempC.append(fl)
tempC = sorted(tempC)   # class:2       val: 0
    
dirE="/home/putus/PROJECT/EEG/setE/"
tempE = []
for file in os.listdir(dirE):
    fl = dirE + file
    tempE.append(fl)
tempE = sorted(tempE) # class: 3        val: 1


# # create small tables 

# In[23]:

tb=[]
st = 'A'
for i in range(len(tempB)):
    x = pd.read_table(tempB[i],header=None)
    x.columns=[st+str(i)]
    tb.append(x)
    
tc=[]
st = 'A'
for i in range(len(tempC)):
    x = pd.read_table(tempC[i],header=None)
    x.columns=[st+str(i)]
    tc.append(x)
    
te=[]
st = 'A'
for i in range(len(tempE)):
    x = pd.read_table(tempE[i],header=None)
    x.columns=[st+str(i)]
    te.append(x)


# # make big table

# In[24]:

def table(table):
    big_table = None
    for ta in table:
        big_table = pd.concat([big_table, ta],axis=1)
    return big_table


# In[25]:

bigB = table(tb)
bigC = table(tc)
bigE = table(te)
head = list(bigB.columns.values)


# In[31]:

len(bigB.columns)


# # creat sub matrix for each

# In[7]:

# creat a matrix
def creat_mat(mat):
    matx = np.zeros((len(mat),(len(head))))
    for i in range(len(head)):
        matx[:,i] = mat[head[i]]
        sleep(0.01)
    return matx
# every mat contain probs mat[:,i] total 28 start form mat1


# In[8]:

matB = creat_mat(bigB) # : refers to healthy data
matC = creat_mat(bigC) # : refers to Inter-ictal (transition between healthy to seizure)
matE = creat_mat(bigE) # : of ictal or seizures

matB = np.nan_to_num(matB) # matB[:,0] --- > channel 0, matB[:,1] --- > channel 1 like that
matC = np.nan_to_num(matC)
matE = np.nan_to_num(matE)


# 4097 data point per channel 
# 173.61 Hz sample rate and there are 4097 data point for each channel
# total 100 channel are their
# 4097/173.61 = 23.59 sec 
# the raw data from one of the channels for the 23.59 sec


# #  <---------  create number of featuters ------------>

# # 7- Features
# 
# # (DFA,HFD,SVD_Entropy,Fisher_Information,ApEn,PFD,Spectral_Entropy)

# In[15]:

# source: https://www.hindawi.com/journals/cin/2011/406391/
def features(mat):
    Kmax = 5
    Tau  = 4
    DE   = 10
    M    = 10
    R    = 0.3
    Band = np.arange(1,86)
    Fs   = 173
    DFA                = pyeeg.dfa(mat)
    HFD                = pyeeg.hfd(mat,Kmax)
    SVD_Entropy        = pyeeg.svd_entropy(mat,Tau,DE)
    Fisher_Information = pyeeg.fisher_info(mat,Tau,DE)
    #ApEn               = pyeeg.ap_entropy(mat, M, R)      # very slow 
    PFD                = pyeeg.pfd(mat)
    Spectral_Entropy   = pyeeg.spectral_entropy(mat, Band, Fs, Power_Ratio=None)
    sleep(0.01)
    
    return (DFA,HFD,SVD_Entropy,Fisher_Information,PFD,Spectral_Entropy)


# # Compute the features of datasets

# In[41]:

# this is three classification proble
# class1: healthy
# class2: transition
# class3: dieases


# In[153]:

# create features of class I
f1_B    = np.zeros((100,1))
f2_B    = np.zeros((100,1))
f3_B    = np.zeros((100,1))
f4_B    = np.zeros((100,1))
f5_B    = np.zeros((100,1))
f6_B    = np.zeros((100,1)) 
cl_B    = np.ones((100,1)) # 1


for i in range(100):
    [f1_B[i,0],f2_B[i,0],f3_B[i,0],f4_B[i,0],f5_B[i,0],f6_B[i,0]]=features(matB[:,i])
    


# In[62]:

# create features of class II
f1_C    = np.zeros((100,1))
f2_C    = np.zeros((100,1))
f3_C    = np.zeros((100,1))
f4_C    = np.zeros((100,1))
f5_C    = np.zeros((100,1))
f6_C    = np.zeros((100,1))
cl_C    = np.zeros((100,1)) # transition means 0

for i in range(100):
    [f1_C[i,0],f2_C[i,0],f3_C[i,0],f4_C[i,0],f5_C[i,0],f6_C[i,0]]=features(matC[:,i])
    


# In[106]:

# create features of class III
f1_E    = np.zeros((100,1))
f2_E    = np.zeros((100,1))
f3_E    = np.zeros((100,1))
f4_E    = np.zeros((100,1))
f5_E    = np.zeros((100,1))
f6_E    = np.zeros((100,1)) 
cl_E    = np.negative(np.ones((100,1))) # -1

for i in range(100):
    [f1_E[i,0],f2_E[i,0],f3_E[i,0],f4_E[i,0],f5_E[i,0],f6_E[i,0]]=features(matE[:,i])
    


# # create Feature and class matrix for each Classes

# In[107]:

MftB = np.concatenate([f1_B,f2_B,f3_B,f4_B,f5_B,f6_B,cl_B],axis=1)
MftC = np.concatenate([f1_C,f2_C,f3_C,f4_C,f5_C,f6_C,cl_C],axis=1)
MftE = np.concatenate([f1_E,f2_E,f3_E,f4_E,f5_E,f6_E,cl_E],axis=1)


# In[115]:

FCM_B = pd.DataFrame(MftB,columns=['f1','f2','f3','f4','f5','f6','class'])
FCM_C = pd.DataFrame(MftC,columns=['f1','f2','f3','f4','f5','f6','class'])
FCM_E = pd.DataFrame(MftE,columns=['f1','f2','f3','f4','f5','f6','class'])
#FCM_B.head(4)


# In[182]:

TotalDataset = pd.concat([FCM_B,FCM_C,FCM_E],ignore_index=False)
#TotalDataset[200:300]


# # Split arrays or matrices into random train and test subsets

# In[122]:

from sklearn.model_selection import train_test_split


# In[159]:

X = TotalDataset[['f1','f2','f3','f4','f5','f6']]
y = TotalDataset[['class']]
X = np.asarray(X)
y = np.asarray(y)


# In[160]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # apply machine learning 

# In[161]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[162]:

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


# In[163]:

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# In[172]:

clf_score=[]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        clf_score.append([score,name])


# In[174]:

print(clf_score)


# In[ ]:



