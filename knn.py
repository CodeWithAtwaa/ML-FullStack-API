from sklearn.neighbors import KNeighborsClassifier
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print("KNN Accuracy:", model.score(X_test, y_test))

# KNN Accuracy: 0.789


# python3 -m venv venv
# source venv/bin/activate