

import pandas as pd
import numpy as np
import os


csv_path = r"C:\Users\sevit\Downloads\appliance energy\Appliances_energy_prediction.csv"  # Change path if needed

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}")

df = pd.read_csv(csv_path)
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


print("\n--- INFO ---")
print(df.info())
print("\nMissing values per column:\n", df.isna().sum())

if 'date' not in df.columns:
    raise ValueError("Expected column 'date' not found. Please check dataset columns.")

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)


df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Cyclic features for hour (sin/cos to capture circularity)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)


for lag in [1, 3, 6, 12]:
    df[f'lag_{lag}'] = df['Appliances'].shift(lag)

# ---------------------------
# Step 6: Rolling Statistics (captures trend/smoothness)
# ---------------------------
df['roll_mean_3'] = df['Appliances'].shift(1).rolling(window=3).mean()
df['roll_std_3'] = df['Appliances'].shift(1).rolling(window=3).std()


df = df.dropna().reset_index(drop=True)


df = df.fillna(df.median(numeric_only=True))
print("\n✅ Missing values handled. Final shape:", df.shape)


df.columns = [c.strip().replace(' ', '_') for c in df.columns]


save_path = r"C:\Users\sevit\Downloads\Appliances_energy_prediction_preprocessed.csv"
df.to_csv(save_path, index=False)
print(f"\n✅ Preprocessed dataset saved successfully at:\n{save_path}")


print("\n--- Final Columns ---")
print(df.columns.tolist())

print("\n--- Sample ---")
print(df.head(5))
