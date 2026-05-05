from sklearn.svm import SVC
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = SVC()
model.fit(X_train, y_train)
print("SVM Accuracy:", model.score(X_test, y_test))
