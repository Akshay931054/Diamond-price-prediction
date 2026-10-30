import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

file_path = r"C:\Users\ANIL KUMAR SINGH\OneDrive\Desktop\Project\Fraud_Data.csv"
data = pd.read_csv(file_path)
print(data.head())

print("\nColumn Names:")
print(data.columns)

'''Removes accidental spaces from column names'''
data.columns = data.columns.str.strip()
for column in CAT_COLS + ID_COLS:
    data[column] = data[column].astype("string").str.strip()