from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)
