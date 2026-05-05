from sklearn.linear_model import LogisticRegression
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression Accuracy:", model.score(X_test, y_test))

# Logistic Regression Accuracy: 0.789