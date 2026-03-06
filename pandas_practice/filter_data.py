import pandas as pd

data = {
    "Name": ["Amit", "Sara", "John", "Riya"],
    "Marks": [85, 90, 78, 92]
}

df = pd.DataFrame(data)

high_marks = df[df["Marks"] > 80]

print("Students scoring above 80:")
print(high_marks)
