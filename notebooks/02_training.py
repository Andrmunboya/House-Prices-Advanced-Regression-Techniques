import pandas as pd
from catboost import CatBoostRegressor

import pandas as pd

data_test = pd.read_csv('../data/test.csv')

features = [
    'MSSubClass', 'MSZoning', 
    'LotFrontage', 'GarageArea', 'GarageQual',
    'WoodDeckSF', 'PoolQC', 'MoSold', 'YrSold',
     'TotRmsAbvGrd', 'BsmtFinSF1', 'BedroomAbvGr',
            ]

X_test = data_test[features]

# Заполняем пустосты

cat_features = ['MSZoning', 'GarageQual', 'PoolQC']

for col in cat_features:
    X_test[col] = X_test[col].astype(str).fillna('missing')

# Загружаем готовые веса

model_cat = CatBoostRegressor()

model_cat.load_model("../model/house_prices_catboost.cbm")
print("модель загруженна")

test_predictions = model_cat.predict(X_test)

# формируем итоговый ответ для kagle

submission = pd.DataFrame({
    'Id': data_test['Id'],
    'SalePrice': test_predictions
})

submission.to_csv("../data/house_prices_submission.csv", index=False)

print('Успешная выгрузка данyых с прогнозами!')
display(submission.head(10))
