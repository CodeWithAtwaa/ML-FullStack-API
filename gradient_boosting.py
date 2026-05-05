from sklearn.ensemble import GradientBoostingClassifier
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
print("Gradient Boosting Accuracy:", model.score(X_test, y_test))
