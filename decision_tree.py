from sklearn.tree import DecisionTreeClassifier
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Decision Tree Accuracy:", model.score(X_test, y_test))
