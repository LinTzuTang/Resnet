from xgboost import XGBClassifier
xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
xgboostModel.fit(X_train, y_train)
predicted = xgboostModel.predict(X_train)