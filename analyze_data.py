import pandas as pd

# Load dataset
df = pd.read_csv("Dataset/fraud_data.csv")

# Count normal and fraudulent transactions
class_counts = df["Class"].value_counts()

print("Transaction Distribution:")
print(class_counts)

print("\nPercentage Distribution:")
print(df["Class"].value_counts(normalize=True) * 100)

print("\nFraudulent Transactions:", class_counts.get(1, 0))
print("Normal Transactions:", class_counts.get(0, 0))