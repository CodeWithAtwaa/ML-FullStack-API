from xgboost import XGBClassifier
from preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
print("XGBoost Accuracy:", model.score(X_test, y_test))
