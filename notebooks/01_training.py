import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


data_train = pd.read_csv('../data/train.csv')

# # Таблица
# display(data_train.head(10))
# data_train.info()

features = [
    'MSSubClass', 'MSZoning', 
    'LotFrontage', 'GarageArea', 'GarageQual',
    'WoodDeckSF', 'PoolQC', 'MoSold', 'YrSold',
     'TotRmsAbvGrd', 'BsmtFinSF1', 'BedroomAbvGr',
            ]

cat_features = ['MSZoning', 'GarageQual', 'PoolQC']

X = data_train[features]

y = data_train['SalePrice']

# Заполняем пустосты
y = y.fillna(0).astype(int)

for col in cat_features:
    X[col] = X[col].astype(str).fillna('missing')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ОБУЧАЕМ МОДЕЛЬ
print('--- НАЧАЛОСЬ ОБУЧЕНИЕ МОДЕЛИ ---')
model_cat = CatBoostRegressor(
    iterations=600,
    learning_rate=0.04,
    depth=6,
    eval_metric='RMSE',
    cat_features=cat_features,
    random_seed=42,
    verbose=50,
)

model_cat.fit(
    X_train, 
    y_train,
    early_stopping_rounds=50, 
    eval_set=(X_val, y_val)
    )
print('--- ОБУЧЕНИЕ ЗАКОНЧЕНО ---')

# Метрики
print('=== РАСЧЕТ МЕТРИК ===')

y_pred = model_cat.predict(X_val)

rmse_score = root_mean_squared_error(y_val, y_pred)
print(f'ИТОГОВЫЙ RMSE: {rmse_score:.2f} $')
r2 = r2_score(y_val, y_pred)

print(f' Кофф детерминации: {r2:.4f}')

model_cat.save_model("../model/house_prices_catboost.cbm")
