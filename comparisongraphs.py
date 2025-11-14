# ============================================================
# 📊 COMPARISON: BEFORE vs AFTER PREPROCESSING (FIXED VERSION)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
sns.set(style="whitegrid", palette="muted")

raw_path = r"C:\Users\sevit\Downloads\appliance energy\Appliances_energy_prediction.csv"
processed_path = r"C:\Users\sevit\Downloads\Appliances_energy_prediction_preprocessed.csv"
# -----------------------------------------------
# Step 1: Load both datasets
# -----------------------------------------------
df_raw = pd.read_csv(raw_path)
df_processed = pd.read_csv(processed_path)

# -----------------------------------------------
# Step 2: Safely convert date columns
# -----------------------------------------------
def safe_to_datetime(series):
    """Safely parse datetime with dayfirst=True and coerce errors."""
    return pd.to_datetime(series, dayfirst=True, errors='coerce')

df_raw['date'] = safe_to_datetime(df_raw['date'])
df_processed['date'] = safe_to_datetime(df_processed['date'])

# Drop rows where date couldn't be parsed
df_raw = df_raw.dropna(subset=['date'])
df_processed = df_processed.dropna(subset=['date'])

# Sort by date
df_raw = df_raw.sort_values('date').reset_index(drop=True)
df_processed = df_processed.sort_values('date').reset_index(drop=True)

print("✅ Raw shape:", df_raw.shape)
print("✅ Processed shape:", df_processed.shape)
print("✅ Date parsing successful!")

# -----------------------------------------------
# Step 3: Energy Consumption Trend - Raw vs Processed
# -----------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(df_raw['date'], df_raw['Appliances'], label='Raw Data', color='gray', alpha=0.6)
plt.plot(df_processed['date'], df_processed['Appliances'], label='Processed Data', color='royalblue', linewidth=1)
plt.title('Comparison of Energy Consumption (Before vs After Processing)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Energy (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# Step 4: Smoothed Trend - Rolling Mean vs Actual
# -----------------------------------------------
if 'roll_mean_3' in df_processed.columns:
    plt.figure(figsize=(12,5))
    plt.plot(df_processed['date'], df_processed['Appliances'], label='Actual (Processed)', color='skyblue', alpha=0.5)
    plt.plot(df_processed['date'], df_processed['roll_mean_3'], label='Rolling Mean (Smoothed)', color='red', linewidth=2)
    plt.title('Smoothed Trend Using roll_mean_3 Feature', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Energy (Wh)')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Note: 'roll_mean_3' column not found in processed dataset.")

# -----------------------------------------------
# Step 5: Distribution Comparison - Before vs After
# -----------------------------------------------
plt.figure(figsize=(10,5))
sns.kdeplot(df_raw['Appliances'], label='Before Processing', color='gray', fill=True, alpha=0.4)
sns.kdeplot(df_processed['Appliances'], label='After Processing', color='blue', fill=True, alpha=0.4)
plt.title('Distribution of Appliance Energy Consumption (Before vs After)', fontsize=14)
plt.xlabel('Energy (Wh)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# Step 6: Hourly Pattern Comparison
# -----------------------------------------------
if 'hour' not in df_raw.columns:
    df_raw['hour'] = pd.to_datetime(df_raw['date']).dt.hour

plt.figure(figsize=(10,4))
sns.barplot(x='hour', y='Appliances', data=df_raw, color='gray', alpha=0.6, label='Before Processing')
sns.barplot(x='hour', y='Appliances', data=df_processed, color='royalblue', alpha=0.7, label='After Processing')
plt.title('Average Energy Consumption by Hour (Before vs After)', fontsize=14)
plt.xlabel('Hour of Day')
plt.ylabel('Average Energy (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# Step 7: Correlation Heatmap Comparison
# -----------------------------------------------
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
sns.heatmap(df_raw.corr(numeric_only=True), cmap='coolwarm', center=0)
plt.title('Before Processing')

plt.subplot(1,2,2)
sns.heatmap(df_processed.corr(numeric_only=True), cmap='coolwarm', center=0)
plt.title('After Processing')

plt.suptitle('Feature Correlation Comparison', fontsize=15)
plt.tight_layout()
plt.show()

# -----------------------------------------------
# Step 8: Zoomed Trend (Optional - for report clarity)
# -----------------------------------------------
sample = df_processed[df_processed['date'].between('2016-01-18', '2016-01-19')]

plt.figure(figsize=(10,4))
plt.plot(sample['date'], sample['Appliances'], label='Processed Data', color='blue', alpha=0.7)
if 'roll_mean_3' in sample.columns:
    plt.plot(sample['date'], sample['roll_mean_3'], label='Rolling Mean', color='red', linewidth=2)
plt.title('Zoomed View (24-Hour Energy Pattern)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Energy (Wh)')
plt.legend()
plt.tight_layout()
plt.show()

print("✅ All comparison graphs generated successfully!")
