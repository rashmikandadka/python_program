import pandas as pd

# Load dataset
df = pd.read_csv("Iris_1.csv")

print(df.head())
print(df.describe())

filtered = df[df["petal_length"] > 5]

print(filtered)
