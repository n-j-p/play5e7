import polars as pl
import pandas as pd
import os
from collections import namedtuple

def load_data(data_path = '/kaggle/input/playground-series-s5e7'):
    train = pl.read_csv(os.path.join(data_path, 'train.csv'))

    comp = pl.read_csv(os.path.join(data_path, 'test.csv'))
    sub = pl.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    Data = namedtuple('Data', ['train',
                               'comp',
                               'sub'])
    return Data(train, 
                comp, 
                sub)


