import pandas as pd

data = {
    "Name": ["Amit", "Sara", "John", "Riya"],
    "Marks": [85, 90, 78, 92]
}

df = pd.DataFrame(data)

print("Data:")
print(df)

print("Average Marks:", df["Marks"].mean())
