import _pickle as cPickle
from Feature_Extractor import Feature_Extractor
from Predictor import Predictor
import pandas as pd
import numpy as np

def train():
    # training file is a corpus of similar questions from Quora
    training_file = 'train.csv'
    print("Creating Feature Extractor instance")
    feature_extractor = Feature_Extractor()
    print("Data Cleaning")
    filtered_data = feature_extractor.Data_Cleaning(training_file)
    print("Feature Set I")
    feature_extractor.Feature_set1(filtered_data)
    print("Feature Set II")
    feature_extractor.Feature_set2(filtered_data)
    print("Words to Vectors")
    feature_extractor.words_to_vectors(filtered_data, training_file)
    features_file = 'Features_File_' + training_file
    print("Creating Predictor instance")
    predictor = Predictor()
    predictor.Xgb_Boost_Model(features_file)

def evaluate():
    f = open('XGBOOST_TRAIN.pkl', 'rb')
    loaded_model = cPickle.load(f)
    dialogues_file = 'dialogues_to_be_evaluated.csv'
    feature_extractor = Feature_Extractor()
    x = "dialogues_to_be_evaluated.csv"
    while(x!="e"):
        dialogues_file = x
        if(dialogues_file):
            filtered_data = feature_extractor.Data_Cleaning(dialogues_file)
            feature_extractor.Feature_set1(filtered_data)
            feature_extractor.Feature_set2(filtered_data)
            feature_extractor.words_to_vectors(filtered_data, dialogues_file)
            features_file = 'Features_File_' + dialogues_file
            df = pd.read_csv(features_file)
            test = np.array(df)
            Features = test[:, 1:628]
            predictions = loaded_model.predict(Features)
            output = pd.read_csv(dialogues_file)
            output['is_related']=predictions
            output.to_csv('Final_Output.csv')
        else:
            print("Invalid input..")
        x = input("Please input a valid csv file with dialogues to evaluate or press e to exit: ")

# Be careful. Training will approximately take about 90 minutes
train()
