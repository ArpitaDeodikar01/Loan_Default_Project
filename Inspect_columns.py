import pandas as pd


# Path to your dataset inside the data folder
data_path = "data\Loan_default.csv"   # Make sure your CSV is named exactly like this


# Load the dataset
df = pd.read_csv(data_path)


print("\n✅ Dataset loaded successfully!\n")
print("First 5 rows:\n", df.head())
print("\nColumn Names:\n", df.columns.tolist())
print("\nInfo:")
print(df.info())
print("\nDescription:")
print(df.describe())

