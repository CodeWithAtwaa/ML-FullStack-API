from sklearn.ensemble import AdaBoostClassifier
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = AdaBoostClassifier()
model.fit(X_train, y_train)
print("AdaBoost Accuracy:", model.score(X_test, y_test))
