#import polars as pl
import pandas as pd
import os
from collections import namedtuple

def load_data(data_path = '/kaggle/input/playground-series-s5e7'):
    train = pd.read_csv(os.path.join(data_path, 'train.csv'),
                        index_col='id')

    comp = pd.read_csv(os.path.join(data_path, 'test.csv'),
                       index_col='id')
    sub = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    Data = namedtuple('Data', ['train',
                               'comp',
                               'sub'])
    return Data(train, 
                comp, 
                sub)


from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
#oe = OrdinalEncoder().set_output(transform='pandas')
pipe = Pipeline([('encode', OrdinalEncoder().set_output(transform='pandas')),
                 ('impute', IterativeImputer(max_iter=100))
                ]).set_output(transform='pandas')
                