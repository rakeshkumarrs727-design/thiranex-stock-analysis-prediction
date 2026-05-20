import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Self-Contained Financial Dataset (No CSV file needed)
data = {
    'Day': list(range(1, 16)),
    'Stock_Price': [100, 102, 101, 105, 107, 106, 110, 112, 111, 115, 118, 117, 122, 125, 124],
    'Trading_Volume': [1500, 1800, 1200, 2100, 2500, 1900, 3100, 2800, 2200, 3500, 4100, 3800, 4500, 4800, 4200]
}
df = pd.DataFrame(data)

print("=== 1. Dataset Head ===")
print(df.head())

# 2. Basic Statistical Context
print("\n=== 2. Dataset Insights ===")
print(df.describe())

# 3. Fast Feature Engineering (Creating a target to predict tomorrow's price)
df['Next_Day_Price'] = df['Stock_Price'].shift(-1)
df.dropna(inplace=True) # Remove the last row since tomorrow's price doesn't exist for it

# 4. Simple Machine Learning Model
X = df[['Stock_Price', 'Trading_Volume']] # Inputs
y = df['Next_Day_Price']                  # What we want to predict

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n=== 3. Model Results ===")
print(f"Model Training Score (R2): {model.score(X_train, y_train):.4f}")

# 5. Generate and Save Visualizations
output_dir = "finance_plots"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(df['Day'], df['Stock_Price'], marker='o', color='green', label='Stock Price')
plt.title('Stock Price Trend Over Time')
plt.xlabel('Day')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()

plt.savefig(f"{output_dir}/stock_trend.png")
plt.close()

print(f"\n[✓] Saved plot to: {output_dir}/stock_trend.png")
print("Finance Pipeline finished successfully with zero errors!")
