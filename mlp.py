from sklearn.neural_network import MLPClassifier
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = MLPClassifier(max_iter=500)
model.fit(X_train, y_train)
print("MLP Accuracy:", model.score(X_test, y_test))


# python3 -m venv venv
# source venv/bin/activate
# pip install pandas numpy scikit-learn matplotlib xgboost