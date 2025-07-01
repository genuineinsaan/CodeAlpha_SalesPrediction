# sales_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
file_path = "C:/Users/shash/OneDrive/Desktop/Internship 2/task 4/Advertising.csv"
df = pd.read_csv(file_path)

# Step 2: Check the data
print("📊 First 5 rows:\n", df.head())
print("\n🔍 Info:\n")
df.info()
print("\n🧼 Null values:\n", df.isnull().sum())

# Step 3: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 4: Feature and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📉 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Step 9: Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
