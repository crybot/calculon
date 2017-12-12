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


class Predictor():
    def Xgb_Boost_Model(self,file):
        PICKLE_FILE_PATH = "XGBOOST_TRAIN" + ".pkl"
        dataframe = pd.read_csv(file)
        dataframe = dataframe.replace(np.nan, -5555555)
        dataframe = dataframe.replace(np.inf, -5555555)
        data = np.array(dataframe)
        target = data[:, -1]
        Features = data[:, 1:628]
        X_train, X_test, y_train, y_test = train_test_split(Features, target, test_size=0.2)

        # Hyper-tuned Parameters of xgboost

        xg_boost = xg.XGBClassifier()

        xg_boost.fit(Features, target)
        predictions = xg_boost.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)
        print ('Accuracy : %.2f%%' % (accuracy *  100))

        with open(PICKLE_FILE_PATH, 'wb') as pickle_file:
            cPickle.dump(xg_boost, pickle_file)
        print("Saved model to PICKLE file")

"""
def Main():
    print("Programe started....")
    file = sys.argv[1]  # File name
    start_time = time.time()
    xgboost_model = Xgb_Boost_Model(file)
    print("--- %s seconds ---" % (time.time() - start_time))


Main()
"""