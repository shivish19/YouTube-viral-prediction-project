"""
Train the viral / views models on USvideos.csv and export them to api/model.json.
Run ONCE locally before deploying:   python train_export.py path/to/USvideos.csv

Only the 5 numeric features are used (likes, dislikes, comment_count, title_length,
days_since_publish). A web form can't sensibly ask for one of ~2,000 channel dummies, and a
numeric-only model is just a handful of numbers, so the Vercel function needs no scikit-learn.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'USvideos.csv'
FEATURES = ['likes', 'dislikes', 'comment_count', 'title_length', 'days_since_publish']

df = pd.read_csv(csv_path)

# same feature engineering as youtubeanalytics.py
threshold = df['views'].quantile(0.75)
df['viral'] = (df['views'] > threshold).astype(int)
publish = pd.to_datetime(df['publish_time'])
trending = pd.to_datetime(df['trending_date'], format='%y.%d.%m', utc=True)
df['days_since_publish'] = (trending - publish.dt.normalize()).dt.days.clip(lower=0)
df['title_length'] = df['title'].str.len()

X = df[FEATURES].astype(float)
X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, df['views'], df['viral'], test_size=0.2, random_state=42, stratify=df['viral'])

scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

lin = LinearRegression().fit(Xtr, yr_train)
log = LogisticRegression(max_iter=1000).fit(Xtr, yc_train)

pred = lin.predict(Xte)
prob = log.predict_proba(Xte)[:, 1]
label = (prob >= 0.5).astype(int)
metrics = {
    'r2': r2_score(yr_test, pred),
    'mae': mean_absolute_error(yr_test, pred),
    'accuracy': accuracy_score(yc_test, label),
    'precision': precision_score(yc_test, label),
    'recall': recall_score(yc_test, label),
    'f1': f1_score(yc_test, label),
    'roc_auc': roc_auc_score(yc_test, prob),
}

model = {
    'features': FEATURES,
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'linear': {'coef': lin.coef_.tolist(), 'intercept': float(lin.intercept_)},
    'logistic': {'coef': log.coef_[0].tolist(), 'intercept': float(log.intercept_[0])},
    'viral_threshold': float(threshold),
    'metrics': {k: float(v) for k, v in metrics.items()},
    'trained_on_rows': int(len(df)),
}
with open('api/model.json', 'w') as f:
    json.dump(model, f, indent=2)

print('Saved api/model.json')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')
