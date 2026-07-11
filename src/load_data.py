import pandas as pd
import numpy as np
import os 

def load_data(load_path=None):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(f'{root_dir}/data/diabetes_prediction_dataset.csv')
    return df