from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Random Forest Accuracy:", model.score(X_test, y_test))
