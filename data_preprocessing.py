# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:41:59 2022

@author: Sharona
"""
# imports
import pandas as pd

# open training data as pandas df + add column names
train_df = pd.read_csv("SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt", sep="\t", header=None, names=["Book", "Sent nr", "Token nr", "Token", "Label"])

# print partial df to check if it went well
print(train_df.head(10))

# add each feature as new column in df
#code for extracting features here

# convert pandas df to .conll file (uncomment when ready to use)
#outputfile = "SEM_train_data.conll"
#train_df.to_csv(outputfile, sep='\t', header=True, quotechar='|', index=False)