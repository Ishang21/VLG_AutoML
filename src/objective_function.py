from sklearn.metrics import accuracy_score


def objective_function(model_class,params):
   
    model = model_class(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy


