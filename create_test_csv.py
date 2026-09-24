import pandas as pd

input_file = "Dataset/fraud_data.csv"
output_file = "Dataset/test_transactions.csv"

# Read first 1000 transactions
df = pd.read_csv(input_file, nrows=1000)

# Save as a smaller test CSV
df.to_csv(output_file, index=False)

print("Test CSV created successfully!")
print("Rows:", len(df))
print("File:", output_file)