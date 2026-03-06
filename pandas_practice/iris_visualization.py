import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Iris_1.csv")

# Scatter plot
plt.scatter(df["sepal_length"], df["petal_length"])

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")

plt.show()
