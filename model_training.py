

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import os

sns.set(style="whitegrid", palette="muted")

csv_path = r"C:\Users\sevit\Downloads\Appliances_energy_prediction_preprocessed.csv"
df = pd.read_csv(csv_path)

df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
print("✅ Dataset loaded. Shape:", df.shape)


if 'humidity' not in df.columns:
    rh_cols = [col for col in df.columns if col.startswith('RH_') and col != 'RH_out']
    if rh_cols:
        df['humidity'] = df[rh_cols].mean(axis=1)
        print(f"✅ Added 'humidity' column (avg of {len(rh_cols)} RH sensors)")
    else:
        raise ValueError("No RH_1...RH_9 columns found to compute humidity!")

if 'hour' not in df.columns:
    df['hour'] = df['date'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    print("✅ Added time-based features: hour, hour_sin, hour_cos")

target = 'Appliances'


features = [
    'T1', 'T2', 'T_out', 'RH_out', 'lights', 'humidity',
    'hour', 'hour_sin', 'hour_cos', 'lag_1', 'lag_3', 'roll_mean_3'
]


missing = [col for col in features if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing features in dataset: {missing}")

X = df[features]
y = df[target]


train_size = int(0.8 * len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    print(f"\n📊 {model_name} Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")

    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


tscv = TimeSeriesSplit(n_splits=5)


models = {
    "Ridge Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=42))
    ])
}

param_grids = {
    "Ridge Regression": {'model__alpha': [0.1, 1.0, 10.0]},
    "Random Forest": {
        'model__n_estimators': [100, 200],
        'model__max_depth': [6, 10, None],
        'model__min_samples_leaf': [2, 5]
    },
    "Gradient Boosting": {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5]
    }
}


results = []
best_models = {}

for name, pipeline in models.items():
    print(f"\n🚀 Training {name}...")
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grids[name],
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model

    print(f"✅ Best Parameters for {name}: {grid.best_params_}")

    y_pred = best_model.predict(X_test)
    result = evaluate_model(y_test, y_pred, name)
    results.append(result)

results_df = pd.DataFrame(results)
print("\n📋 Model Performance Summary:")
print(results_df)


best_model_name = results_df.sort_values('RMSE').iloc[0]['Model']
best_model = best_models[best_model_name]

file_name = f"best_energy_model_{best_model_name.replace(' ', '_')}.joblib"
joblib.dump(best_model, os.path.join(os.getcwd(), file_name))

print(f"\n💾 Best model saved as: {file_name}")
print(f"✅ Model trained with features: {features}")


plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='RMSE', data=results_df, palette='coolwarm')
plt.title('Model Comparison - RMSE')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='R2', data=results_df, palette='viridis')
plt.title('Model Comparison - R² Score')
plt.ylabel('R²')
plt.tight_layout()
plt.show()


y_pred_best = best_model.predict(X_test)
plt.figure(figsize=(12,5))
plt.plot(df['date'].iloc[train_size:], y_test.values, label='Actual', color='gray', alpha=0.7)
plt.plot(df['date'].iloc[train_size:], y_pred_best, label=f'Predicted ({best_model_name})', color='red')
plt.title(f'Actual vs Predicted Energy Consumption - {best_model_name}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Energy (Wh)')
plt.legend()
plt.tight_layout()
plt.show()


residuals = y_test - y_pred_best
plt.figure(figsize=(8,4))
sns.histplot(residuals, bins=40, kde=True, color='purple')
plt.title(f'Residual Distribution - {best_model_name}', fontsize=14)
plt.xlabel('Prediction Error (Wh)')
plt.tight_layout()
plt.show()
