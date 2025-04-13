from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Load your dataset
df = pd.read_csv(r'C:\Users\mbsfa\OneDrive\Desktop\SSN\SEM6\ML_LAB\MINI_PROJECT\stock_data.csv')

# Define Features & Target
features = ['Stock_2', 'Stock_3', 'Stock_4', 'Stock_5']
target = 'Stock_1'

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = BaggingRegressor()
model.fit(X_train_scaled, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Evaluate on test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics
metrics = {'mse': mse, 'r2': r2}
with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print(f"Model saved with MSE: {mse:.4f} and R²: {r2:.4f}")

# Save Model
with open('bagging_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save Scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler Pickle Files Generated Successfully!")
