from sklearn.naive_bayes import GaussianNB
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = GaussianNB()
model.fit(X_train, y_train)
print("Naive Bayes Accuracy:", model.score(X_test, y_test))
