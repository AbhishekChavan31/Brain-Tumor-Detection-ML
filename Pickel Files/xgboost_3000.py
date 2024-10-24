
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import pickle


df = pd.read_csv('Zernike_Moments_YN_3000.csv', header=None)


df = df.sample(frac=1, random_state=0).reset_index(drop=True)


df[289].replace(['YES','NO'], [1,0], inplace=True)


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


xgb = XGBClassifier()


xgb.fit(x_train, y_train)


filename = 'xgboost_3000.pkl'
pickle.dump(xgb, open(filename, 'wb'))
