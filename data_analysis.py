import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Display basic information
print("First five records:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Calculate total revenue
total_revenue = data["Revenue"].sum()
print("\nTotal Revenue:", total_revenue)

# Revenue by product category
category_revenue = data.groupby("Product_Category")["Revenue"].sum()

print("\nRevenue by Product Category:")
print(category_revenue)

# Visualization
category_revenue.plot(kind="bar")

plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("sales_by_category.png")
plt.show()

# Revenue by region
region_revenue = data.groupby("Region")["Revenue"].sum()

print("\nRevenue by Region:")
print(region_revenue)

region_revenue.plot(kind="bar")

plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("sales_by_region.png")
plt.show()

# Linear regression
X = data[["Advertising_Spend"]]
y = data["Revenue"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict(X)

print("\nLinear Regression Results")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R-squared:", model.score(X, y))

# Regression visualization
plt.scatter(data["Advertising_Spend"], data["Revenue"])
plt.plot(data["Advertising_Spend"], prediction)

plt.title("Advertising Spend vs Revenue")
plt.xlabel("Advertising Spend")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("regression_result.png")
plt.show()
