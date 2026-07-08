import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE=42

def train(processor, X, y):
    random_forest = Pipeline([
        ("preprocessing", processor),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            bootstrap=False,
            class_weight='balanced',
            criterion='gini',
            max_depth=10,
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=2,
            n_jobs=-1))
    ])

    random_forest.fit(X, y)
    return random_forest
