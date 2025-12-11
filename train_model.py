import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🔄 Training Master Model (UK Standard)...")

# 1. Load Data
df = pd.read_csv('data/used_cars_combined.csv')

# 2. Prepare Features
X = df.drop('price', axis=1)
y = df['price']
X = pd.get_dummies(X, columns=['brand', 'model', 'transmission', 'fuelType'])

# 3. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save Everything
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')
df.sample(2000).to_csv('data/reference_data.csv', index=False)

print("✅ Success! 'car_price_model.pkl' created.")