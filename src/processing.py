import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def processor():
    numeric = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorial = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
    multi_category= ["gender", "smoking_history"]

    processor = ColumnTransformer(
        transformers=[
            ("category", OrdinalEncoder(), multi_category)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return processor