import pandas as pd
import  numpy as np
import os
from sklearn.feature_selection import VarianceThreshold

datafile="D:/Pythonsjwjrm/samplecode/adult.data"
adult=pd.read_csv(datafile,header=None,names=["Age", "Work-Class", "fnlwgt",
"Education", "Education-Num",
"Marital-Status", "Occupation",
"Relationship", "Race", "Sex",
"Capital-gain", "Capital-loss",
"Hours-per-week", "Native-Country",
"Earnings-Raw"])

#adult.dropna( how='all', inplace=True)

adult["LongHours"]=adult["Hours-per-week"]>40

X=np.arange(30).reshape((10,3))

X[:,1]=1

vt=VarianceThreshold()

x=vt.fit_transform(X)

print(x)
