import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import xgboost as xg
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import GridSearchCV
import _pickle as cPickle
import sys


def Xgb_Boost_Model(file):
    PICKLE_FILE_PATH = "XGBOOST_TRAIN" + ".pkl"
    dataframe = pd.read_csv(file)
    dataframe = dataframe.replace(np.nan, -5555555)
    dataframe = dataframe.replace(np.inf, -5555555)
    data = np.array(dataframe)
    target = data[:, -1]
    Features = data[:, 0:628]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    #print(X_train.head())
    #print(X_test.head())
    # Hyper-tuned Parameters of xgboost

    xg_boost = xg.XGBClassifier()

    xg_boost.fit(Features, target)
    predictions = xg_boost.predict(X_test)
    print(predictions)

    with open(PICKLE_FILE_PATH, 'wb') as pickle_file:
        cPickle.dump(xg_boost, pickle_file)


def Main():
    print("Programe started....")
    file = sys.argv[1]  # File name
    start_time = time.time()
    xgboost_model = Xgb_Boost_Model(file)
    print("--- %s seconds ---" % (time.time() - start_time))


Main()
