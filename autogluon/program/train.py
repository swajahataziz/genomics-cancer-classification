#!/usr/bin/env python

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import os
import json
import pickle
import sys
import traceback

import pandas as pd

from sklearn import tree
from tqdm import tqdm

import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

import autogluon as ag
from autogluon import TabularPrediction as task

import time, pickle
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# These are the paths to where SageMaker mounts interesting things in your container.
from autogluon.task.tabular_prediction import TabularDataset

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name='training'
training_path = os.path.join(input_path, channel_name)

# The function to execute the training.
def train():
    print('Starting the training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        with open(param_path, 'r') as tc:
            trainingParams = json.load(tc)

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [ os.path.join(training_path, file) for file in os.listdir(training_path) ]
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))

        print('Found x number of files')        
        all_model_df = pd.read_csv(os.path.join(training_path,'all_model_df.csv'))
        y5 = np.load(os.path.join(training_path,'y5.npy'))
        

        kf = StratifiedKFold(n_splits = 5)

        f = 0

        # for each cancer in tcga
        for c in tqdm(np.unique(y5)): 
            for train_index, test_index in kf.split(all_model_df,y5):
                t1 = time.time()
                print(c,"starting fold",f)
                
                # load prev.pickle with most important biomarkers
                with open(os.path.join(training_path,"c"+str(c)+"_f"+str(f)+"_5hsic5adasynlgbm100ft.b"), "rb") as fp: 
                    train_index,test_index,chsicpredictor,predy,acc = pickle.load(fp)
                
                c_idx = np.where(y5==c)[0]
                cy = np.zeros_like(y5)
                cy[c_idx] = 1
                
                # train an ensemble model with AutoML to maximize accuracy
                train_data = all_model_df.iloc[train_index].iloc[:, chsicpredictor.hsic_idx_]
                train_data["label"] = cy[train_index]
                clf = task.fit(train_data, label="label", presets='best_quality', auto_stack=True, output_directory="_autogluon_c_"+str(c)+"_f"+str(f))
                
                test_y = y5[test_index]
                c_idx = np.where(test_y==c)[0]
                test_y = np.zeros_like(test_y)
                test_y[c_idx] = 1
                
                bpredy = clf.predict(all_model_df.iloc[test_index].iloc[:, chsicpredictor.hsic_idx_])
                bacc = accuracy_score(test_y, bpredy)
                print("done in ",time.time()-t1,"acc",acc) 
                
                # save the results
                with open(os.path.join(model_path,"AutoML_c"+str(c)+"_f"+str(f)+"_5hsic5adasynlgbm100ft.b"), "wb") as fp: 
                    pickle.dump((train_index,test_index,chsicpredictor,predy,acc,bpredy,bacc,clf),fp)
                    
                f+=1

        print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)